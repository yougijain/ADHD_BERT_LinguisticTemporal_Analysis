"""Fetch a real, timestamped, engagement-labelled corpus from Stack Exchange.

The project needs a corpus with three things: prose, an engagement signal, and a
real posting timestamp. It also needs a licence you can defend out loud, because
"I found it on Kaggle" is not a provenance story -- a large share of the Reddit
dumps there were scraped against the platform's terms and redistributed with no
licence at all, and a reviewer who asks where the data came from deserves a
better answer than a dead link.

Stack Exchange is the defensible option:

  * Every question carries `body` (prose), `score` (net votes, and genuinely
    two-sided -- unlike platforms where posts start at 1 and floor at 0), and
    `creation_date` (Unix epoch, UTC).
  * All user contributions are licensed CC BY-SA 4.0. Redistribution is
    explicitly permitted with attribution, which this module records in a
    provenance sidecar next to every CSV it writes.
  * The API is public, documented, and free. No key is needed for the volumes
    this project uses, and the anonymous quota is 300 requests/day, which at
    100 questions per request is 30,000 posts.

Pick a site with enough traffic to have a real hourly pattern and enough
subject variety that the text is not all boilerplate. `stackoverflow` is the
obvious default; the smaller sites (`scifi`, `cooking`, `worldbuilding`) have
more discursive prose and a different audience clock, which makes them a good
second run.

    python -m data.fetch_dataset --site stackoverflow --rows 20000 \\
        --from 2023-01-01 --to 2024-01-01 --out datasets/posts.csv

Nothing here is imported by the training pipeline -- fetching is a separate,
deliberate step, so a training run can never silently hit the network.
"""

import argparse
import gzip
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from training.config import DATASET_DIR

API_BASE = "https://api.stackexchange.com/2.3"
# `withbody` is a built-in filter alias; the default filter omits post bodies
# entirely, which would leave us with titles and nothing else.
API_FILTER = "withbody"
LICENCE = "CC BY-SA 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
TERMS_URL = "https://stackoverflow.com/legal/terms-of-service/public"

# The API is hard-capped at 100 items per request.
MAX_PAGE_SIZE = 100


class _TextExtractor(HTMLParser):
    """Flatten Stack Exchange's HTML bodies to prose.

    Bodies come back as HTML. The pipeline's cleaner strips punctuation it does
    not recognise, which would turn every `<p>` into a stray `p` token, so the
    tags come off here instead.

    `<code>` and `<pre>` contents are dropped by default. On a programming site
    they are most of the character mass, and a character n-gram model handed a
    stack trace will happily learn to predict engagement from variable names.
    That is a leak dressed up as a feature: keep the prose, drop the code, and
    say so. `keep_code=True` is there for when the code really is the subject.
    """

    _SKIP_TAGS = {"code", "pre", "script", "style"}

    def __init__(self, keep_code=False):
        super().__init__(convert_charrefs=True)
        self.keep_code = keep_code
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if not self.keep_code and tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in ("p", "br", "li", "div", "blockquote"):
            self._parts.append(" ")

    def handle_endtag(self, tag):
        if not self.keep_code and tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in ("p", "li", "div", "blockquote"):
            self._parts.append(" ")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self):
        return re.sub(r"\s+", " ", "".join(self._parts)).strip()


def strip_html(markup, keep_code=False):
    """Return the prose content of an HTML fragment. Never raises."""
    if not isinstance(markup, str) or not markup.strip():
        return ""
    parser = _TextExtractor(keep_code=keep_code)
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup must not kill a fetch
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()
    return parser.text()


def _default_transport(url, timeout=30):
    """GET `url` and return the decoded JSON body.

    The Stack Exchange API gzips every response whether or not you ask, and
    urllib does not decompress transparently, so that is handled here.

    Injected as a parameter everywhere below so the tests can exercise the
    paging, backoff, and normalisation logic without a network.
    """
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip" or payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
    return json.loads(payload.decode("utf-8"))


def _to_epoch(value):
    """Parse a YYYY-MM-DD date (or an epoch int) into UTC epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return int(datetime.strptime(text, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp())


def fetch_questions(site="stackoverflow", target_rows=20000, from_date=None,
                    to_date=None, api_key=None, transport=_default_transport,
                    sleep=time.sleep, verbose=True):
    """Page backwards through a site's questions, newest first.

    Walking a time cursor rather than incrementing `page` is deliberate. Deep
    paging on this API is both capped and unstable -- new posts arriving during
    a long fetch shift every later page, so you silently duplicate some rows and
    miss others. Anchoring each request to `todate = oldest_seen - 1` makes the
    window strictly shrink, so the walk is reproducible and gap-free.

    Args:
        site (str): Stack Exchange site key, e.g. "stackoverflow", "cooking".
        target_rows (int): Stop once this many questions have been collected.
        from_date (str | int | None): Oldest post to accept (YYYY-MM-DD).
        to_date (str | int | None): Newest post to accept (YYYY-MM-DD).
        api_key (str | None): Optional key; raises the daily quota from 300.
        transport (callable): url -> parsed JSON. Swapped out in tests.
        sleep (callable): seconds -> None. Swapped out in tests.
    Returns:
        tuple[list[dict], dict]: raw question dicts and a fetch-stats dict.
    Raises:
        RuntimeError: If the API reports an error or the quota is exhausted.
    """
    from_epoch = _to_epoch(from_date)
    cursor = _to_epoch(to_date)

    items, requests_made, quota_remaining = [], 0, None
    seen_ids = set()

    while len(items) < target_rows:
        params = {
            "site": site,
            "order": "desc",
            "sort": "creation",
            "pagesize": MAX_PAGE_SIZE,
            "filter": API_FILTER,
        }
        if from_epoch is not None:
            params["fromdate"] = from_epoch
        if cursor is not None:
            params["todate"] = cursor
        if api_key:
            params["key"] = api_key

        url = f"{API_BASE}/questions?{urllib.parse.urlencode(params)}"
        payload = transport(url)
        requests_made += 1

        if payload.get("error_id") or payload.get("error_message"):
            raise RuntimeError(
                f"Stack Exchange API error {payload.get('error_id')}: "
                f"{payload.get('error_message')}"
            )

        quota_remaining = payload.get("quota_remaining", quota_remaining)
        batch = payload.get("items", [])
        if not batch:
            if verbose:
                print("  No more questions in range.")
            break

        # Dedupe as we go, and within the batch as well as across batches --
        # a question edited mid-walk can be returned twice by the same request.
        for question in batch:
            question_id = question.get("question_id")
            if question_id in seen_ids:
                continue
            seen_ids.add(question_id)
            items.append(question)

        oldest = min(q["creation_date"] for q in batch)
        if verbose:
            stamp = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  {len(items):>6} questions | back to {stamp} | "
                  f"quota left {quota_remaining}")

        # The API asks callers to wait when it is under load. Ignoring it is how
        # you get rate-limited out of a dataset you are halfway through.
        if payload.get("backoff"):
            backoff = float(payload["backoff"])
            if verbose:
                print(f"  API requested a {backoff:g}s backoff; waiting.")
            sleep(backoff)

        if not payload.get("has_more", False):
            break
        if quota_remaining is not None and quota_remaining <= 1:
            raise RuntimeError(
                f"Stack Exchange daily quota exhausted after {requests_made} "
                f"requests ({len(items)} questions collected). Resume tomorrow, "
                "or register a free API key and pass --api-key to raise it."
            )

        cursor = oldest - 1
        if from_epoch is not None and cursor < from_epoch:
            break

    stats = {
        "requests_made": requests_made,
        "quota_remaining": quota_remaining,
        "questions_collected": len(items),
    }
    return items[:target_rows], stats


def to_frame(items, keep_code=False):
    """Normalise raw API questions into the pipeline's CSV schema.

    Returns a frame with `selftext`, `score`, `created_utc` and `title` -- the
    names every downstream module expects -- plus the secondary engagement
    signals, which are not used as the label but are worth having on disk.
    """
    rows = []
    for item in items:
        rows.append({
            "id": item.get("question_id"),
            "title": html.unescape(item.get("title", "") or ""),
            "selftext": strip_html(item.get("body", ""), keep_code=keep_code),
            "score": item.get("score", 0),
            "answer_count": item.get("answer_count", 0),
            "view_count": item.get("view_count", 0),
            "created_utc": item.get("creation_date"),
            "tags": "|".join(item.get("tags", []) or []),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.dropna(subset=["created_utc"])
    return frame.sort_values("created_utc").reset_index(drop=True)


def write_provenance(path, site, frame, stats, from_date, to_date, keep_code):
    """Record where the data came from, next to the data.

    A CSV with no provenance is a liability: six months later nobody can say
    what it is, whether it may be redistributed, or how to reproduce it. This
    sidecar answers all three, and the README quotes it.
    """
    path = Path(path)
    span = {}
    if not frame.empty:
        stamps = pd.to_datetime(frame["created_utc"], unit="s", utc=True)
        span = {
            "first_post_utc": stamps.min().isoformat(),
            "last_post_utc": stamps.max().isoformat(),
        }

    provenance = {
        "source": "Stack Exchange API v2.3",
        "site": site,
        "endpoint": f"{API_BASE}/questions",
        "api_filter": API_FILTER,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "terms_url": TERMS_URL,
        "attribution": (
            f"Content from the Stack Exchange '{site}' site, retrieved via the "
            f"public Stack Exchange API and licensed under {LICENCE} "
            f"({LICENCE_URL}). Individual posts remain attributable to their "
            "authors via the `id` column."
        ),
        "query": {
            "from_date": from_date,
            "to_date": to_date,
            "order": "desc",
            "sort": "creation",
            "page_size": MAX_PAGE_SIZE,
            "code_blocks": "kept" if keep_code else "stripped",
        },
        "fetched_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "fetch_stats": stats,
        **span,
    }

    path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def provenance_path(csv_path):
    """Sidecar path for a dataset CSV: posts.csv -> posts.provenance.json."""
    csv_path = Path(csv_path)
    return csv_path.with_suffix(".provenance.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch a real timestamped corpus from the Stack Exchange API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--site", default="stackoverflow",
                        help="Stack Exchange site key (stackoverflow, cooking, scifi...).")
    parser.add_argument("--rows", type=int, default=20000,
                        help="How many questions to collect.")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Oldest post to accept, YYYY-MM-DD.")
    parser.add_argument("--to", dest="to_date", default=None,
                        help="Newest post to accept, YYYY-MM-DD.")
    parser.add_argument("--out", default=None,
                        help="Output CSV path. Defaults to datasets/<site>_questions.csv.")
    parser.add_argument("--api-key", default=None,
                        help="Optional Stack Apps key; raises the daily quota.")
    parser.add_argument("--keep-code", action="store_true",
                        help="Keep <code>/<pre> blocks in the body. Off by default: "
                             "on a programming site they dominate the character "
                             "n-grams and the baseline learns identifiers.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out) if args.out else DATASET_DIR / f"{args.site}_questions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching up to {args.rows} questions from '{args.site}'...")
    items, stats = fetch_questions(
        site=args.site,
        target_rows=args.rows,
        from_date=args.from_date,
        to_date=args.to_date,
        api_key=args.api_key,
    )

    frame = to_frame(items, keep_code=args.keep_code)
    if frame.empty:
        print("No questions returned. Check --site, --from and --to.")
        return None

    frame.to_csv(out, index=False)
    provenance = write_provenance(
        provenance_path(out), args.site, frame, stats,
        args.from_date, args.to_date, args.keep_code,
    )

    print(f"\nWrote {len(frame)} rows to {out}")
    print(f"  span:    {provenance.get('first_post_utc')} .. "
          f"{provenance.get('last_post_utc')}")
    print(f"  score:   min {frame['score'].min()} median "
          f"{frame['score'].median():.0f} max {frame['score'].max()}")
    print(f"  licence: {LICENCE} -- provenance written to {provenance_path(out)}")
    print("\nNext:")
    print(f"  python main.py --dataset {out} --epochs 3")
    return out


if __name__ == "__main__":
    main()
