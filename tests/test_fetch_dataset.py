"""Fetcher tests. The transport is injected, so nothing here touches the network.

The logic worth testing is the paging walk -- it is the part that quietly
duplicates or drops rows when it is wrong, and a broken fetch produces a CSV
that looks perfectly fine.
"""

import json

import pytest

from data.fetch_dataset import (
    fetch_questions,
    provenance_path,
    strip_html,
    to_frame,
    write_provenance,
    _to_epoch,
)

DAY = 86400
BASE_TS = 1700000000


def _question(question_id, created, score=1, body="<p>Some prose here.</p>"):
    return {
        "question_id": question_id,
        "title": "A title",
        "body": body,
        "score": score,
        "answer_count": 2,
        "view_count": 30,
        "creation_date": created,
        "tags": ["python", "pandas"],
    }


def _fake_api(pages):
    """Return a transport that serves `pages` in order and records the URLs."""
    calls = []

    def transport(url, timeout=30):
        calls.append(url)
        return pages[min(len(calls) - 1, len(pages) - 1)]

    transport.calls = calls
    return transport


class TestStripHtml:
    def test_removes_tags_and_keeps_prose(self):
        # Inline tags close up; only block-level ones become a space.
        assert strip_html("<p>Hello <b>there</b>.</p>") == "Hello there."

    def test_drops_code_blocks_by_default(self):
        markup = "<p>Why does this fail?</p><pre><code>import os</code></pre>"
        text = strip_html(markup)
        assert "Why does this fail?" in text
        assert "import os" not in text

    def test_keep_code_retains_them(self):
        markup = "<p>Why?</p><pre><code>import os</code></pre>"
        assert "import os" in strip_html(markup, keep_code=True)

    def test_decodes_entities(self):
        assert "a < b & c" in strip_html("<p>a &lt; b &amp; c</p>")

    def test_collapses_whitespace(self):
        assert strip_html("<p>a</p>\n\n   <p>b</p>") == "a b"

    def test_empty_and_non_string_inputs(self):
        assert strip_html("") == ""
        assert strip_html(None) == ""
        assert strip_html(123) == ""

    def test_malformed_markup_does_not_raise(self):
        assert "text" in strip_html("<p>text<<<unclosed")


class TestToEpoch:
    def test_parses_iso_date_as_utc(self):
        # 2023-01-01T00:00:00Z
        assert _to_epoch("2023-01-01") == 1672531200

    def test_passes_through_epoch_int_and_digit_string(self):
        assert _to_epoch(1672531200) == 1672531200
        assert _to_epoch("1672531200") == 1672531200

    def test_none_is_none(self):
        assert _to_epoch(None) is None


class TestFetchQuestions:
    def test_collects_items_across_pages(self):
        page_one = {"items": [_question(i, BASE_TS - i * DAY) for i in range(3)],
                    "has_more": True, "quota_remaining": 100}
        page_two = {"items": [_question(10 + i, BASE_TS - (10 + i) * DAY)
                              for i in range(2)],
                    "has_more": False, "quota_remaining": 99}
        items, stats = fetch_questions(
            target_rows=100, transport=_fake_api([page_one, page_two]),
            sleep=lambda _: None, verbose=False,
        )
        assert len(items) == 5
        assert stats["requests_made"] == 2
        assert stats["quota_remaining"] == 99

    def test_cursor_moves_strictly_backwards(self):
        page_one = {"items": [_question(i, BASE_TS - i * DAY) for i in range(3)],
                    "has_more": True, "quota_remaining": 100}
        page_two = {"items": [], "has_more": False, "quota_remaining": 99}
        transport = _fake_api([page_one, page_two])
        fetch_questions(target_rows=100, transport=transport,
                        sleep=lambda _: None, verbose=False)

        # Second request must be anchored at (oldest seen - 1), not page=2.
        oldest = BASE_TS - 2 * DAY
        assert f"todate={oldest - 1}" in transport.calls[1]
        assert "page=" not in transport.calls[1]

    def test_duplicate_ids_are_dropped(self):
        # A post arriving mid-walk can push the same question into two windows.
        repeated = {"items": [_question(1, BASE_TS), _question(1, BASE_TS)],
                    "has_more": False, "quota_remaining": 100}
        items, _ = fetch_questions(target_rows=100, transport=_fake_api([repeated]),
                                   sleep=lambda _: None, verbose=False)
        assert len(items) == 1

    def test_stops_at_target_rows(self):
        page = {"items": [_question(i, BASE_TS - i * DAY) for i in range(50)],
                "has_more": True, "quota_remaining": 100}
        items, _ = fetch_questions(target_rows=10, transport=_fake_api([page]),
                                   sleep=lambda _: None, verbose=False)
        assert len(items) == 10

    def test_stops_at_from_date(self):
        page = {"items": [_question(i, BASE_TS - i * DAY) for i in range(3)],
                "has_more": True, "quota_remaining": 100}
        transport = _fake_api([page])
        fetch_questions(
            target_rows=1000, from_date=BASE_TS - DAY, transport=transport,
            sleep=lambda _: None, verbose=False,
        )
        assert len(transport.calls) == 1

    def test_empty_first_page_returns_nothing(self):
        empty = {"items": [], "has_more": False, "quota_remaining": 100}
        items, stats = fetch_questions(target_rows=100, transport=_fake_api([empty]),
                                       sleep=lambda _: None, verbose=False)
        assert items == []
        assert stats["questions_collected"] == 0

    def test_backoff_is_honoured(self):
        slept = []
        page_one = {"items": [_question(1, BASE_TS)], "has_more": True,
                    "backoff": 2.5, "quota_remaining": 100}
        page_two = {"items": [], "has_more": False, "quota_remaining": 100}
        fetch_questions(
            target_rows=100, transport=_fake_api([page_one, page_two]),
            sleep=slept.append, verbose=False,
        )
        assert slept == [2.5]

    def test_api_error_raises(self):
        error = {"error_id": 502, "error_message": "throttle violation"}
        with pytest.raises(RuntimeError, match="throttle violation"):
            fetch_questions(transport=_fake_api([error]), sleep=lambda _: None,
                            verbose=False)

    def test_exhausted_quota_raises_with_a_resume_hint(self):
        page = {"items": [_question(1, BASE_TS)], "has_more": True,
                "quota_remaining": 0}
        with pytest.raises(RuntimeError, match="quota exhausted"):
            fetch_questions(target_rows=100, transport=_fake_api([page]),
                            sleep=lambda _: None, verbose=False)

    def test_api_key_is_passed_through(self):
        page = {"items": [], "has_more": False, "quota_remaining": 100}
        transport = _fake_api([page])
        fetch_questions(api_key="abc123", transport=transport,
                        sleep=lambda _: None, verbose=False)
        assert "key=abc123" in transport.calls[0]

    def test_site_is_passed_through(self):
        page = {"items": [], "has_more": False, "quota_remaining": 100}
        transport = _fake_api([page])
        fetch_questions(site="cooking", transport=transport,
                        sleep=lambda _: None, verbose=False)
        assert "site=cooking" in transport.calls[0]


class TestToFrame:
    def test_produces_the_canonical_columns(self):
        frame = to_frame([_question(1, BASE_TS)])
        for column in ("selftext", "score", "created_utc", "title"):
            assert column in frame.columns

    def test_sorts_chronologically(self):
        items = [_question(1, BASE_TS), _question(2, BASE_TS - DAY)]
        frame = to_frame(items)
        assert frame["created_utc"].is_monotonic_increasing

    def test_bodies_are_plain_text(self):
        frame = to_frame([_question(1, BASE_TS, body="<p>Hello <i>world</i></p>")])
        assert "<" not in frame["selftext"].iloc[0]

    def test_tags_are_flattened(self):
        assert to_frame([_question(1, BASE_TS)])["tags"].iloc[0] == "python|pandas"

    def test_empty_input_gives_empty_frame(self):
        assert to_frame([]).empty

    def test_rows_without_a_timestamp_are_dropped(self):
        broken = _question(1, BASE_TS)
        broken["creation_date"] = None
        assert len(to_frame([broken, _question(2, BASE_TS)])) == 1

    def test_negative_scores_survive(self):
        # Two-sided scores are the reason this corpus was chosen; clamping them
        # would throw away half the label signal.
        frame = to_frame([_question(1, BASE_TS, score=-4)])
        assert frame["score"].iloc[0] == -4


class TestProvenance:
    def test_sidecar_path_sits_next_to_the_csv(self):
        assert provenance_path("datasets/posts.csv").name == "posts.provenance.json"

    def test_records_licence_and_span(self, tmp_path):
        frame = to_frame([_question(1, BASE_TS), _question(2, BASE_TS + DAY)])
        out = tmp_path / "posts.provenance.json"
        record = write_provenance(
            out, "stackoverflow", frame, {"requests_made": 1},
            "2023-01-01", "2024-01-01", keep_code=False,
        )
        written = json.loads(out.read_text())

        assert written == record
        assert record["licence"] == "CC BY-SA 4.0"
        assert record["rows"] == 2
        assert record["query"]["code_blocks"] == "stripped"
        assert "stackoverflow" in record["attribution"]
        assert record["first_post_utc"] < record["last_post_utc"]

    def test_empty_frame_omits_the_span(self, tmp_path):
        record = write_provenance(
            tmp_path / "p.json", "cooking", to_frame([]), {}, None, None, False,
        )
        assert record["rows"] == 0
        assert "first_post_utc" not in record
