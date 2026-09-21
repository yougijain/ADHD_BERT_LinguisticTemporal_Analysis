"""Column-mapping tests. The point of this layer is that it never guesses over
a column you already have, so most of these assert what it refuses to do."""

import pandas as pd
import pytest

from data.schema import (
    describe_schema,
    infer_column_map,
    normalize_columns,
    parse_column_map,
)


class TestParseColumnMap:
    def test_empty_spec_is_empty_map(self):
        assert parse_column_map("") == {}
        assert parse_column_map(None) == {}

    def test_single_pair(self):
        assert parse_column_map("selftext=body") == {"selftext": "body"}

    def test_multiple_pairs_and_whitespace(self):
        spec = " selftext=body , created_utc = creation_date "
        assert parse_column_map(spec) == {
            "selftext": "body",
            "created_utc": "creation_date",
        }

    def test_trailing_comma_is_tolerated(self):
        assert parse_column_map("score=points,") == {"score": "points"}

    def test_source_may_contain_equals(self):
        assert parse_column_map("selftext=a=b") == {"selftext": "a=b"}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="Malformed"):
            parse_column_map("selftext")

    def test_unknown_canonical_raises(self):
        with pytest.raises(ValueError, match="Unknown canonical column"):
            parse_column_map("upvotes=score")

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="No source column"):
            parse_column_map("selftext=")


class TestInferColumnMap:
    def test_infers_stack_exchange_names(self):
        frame = pd.DataFrame(columns=["body", "score", "creation_date", "title"])
        assert infer_column_map(frame) == {
            "selftext": "body",
            "created_utc": "creation_date",
        }

    def test_leaves_canonical_columns_alone(self):
        frame = pd.DataFrame(columns=["selftext", "score", "created_utc"])
        assert infer_column_map(frame) == {}

    def test_never_overrides_an_existing_canonical_column(self):
        # `score` is present, so `points` must not be picked up -- silently
        # relabelling the target is how you train on the wrong thing.
        frame = pd.DataFrame(columns=["selftext", "score", "points", "created_utc"])
        assert "score" not in infer_column_map(frame)

    def test_alias_matching_is_case_insensitive(self):
        frame = pd.DataFrame(columns=["Body", "Score", "Created_At"])
        inferred = infer_column_map(frame)
        assert inferred["selftext"] == "Body"
        assert inferred["created_utc"] == "Created_At"

    def test_first_listed_alias_wins(self):
        frame = pd.DataFrame(columns=["body", "text", "score", "created_utc"])
        assert infer_column_map(frame)["selftext"] == "body"


class TestNormalizeColumns:
    def _frame(self, **columns):
        return pd.DataFrame({k: [v] for k, v in columns.items()})

    def test_canonical_frame_passes_through(self):
        frame = self._frame(selftext="hello", score=3, created_utc=1700000000)
        out = normalize_columns(frame, verbose=False)
        assert list(out.columns) == ["selftext", "score", "created_utc"]

    def test_inference_renames_in_place(self):
        frame = self._frame(body="hello", score=3, creation_date=1700000000)
        out = normalize_columns(frame, verbose=False)
        assert out["selftext"].iloc[0] == "hello"
        assert out["created_utc"].iloc[0] == 1700000000

    def test_explicit_map_beats_inference(self):
        frame = self._frame(body="wrong", summary="right", score=1,
                            created_utc=1700000000)
        out = normalize_columns(frame, "selftext=summary", verbose=False)
        assert out["selftext"].iloc[0] == "right"

    def test_explicit_map_accepts_a_dict(self):
        frame = self._frame(summary="right", score=1, created_utc=1700000000)
        out = normalize_columns(frame, {"selftext": "summary"}, verbose=False)
        assert out["selftext"].iloc[0] == "right"

    def test_renaming_onto_an_occupied_name_does_not_duplicate(self):
        # Mapping score=points while a `score` column already exists must leave
        # exactly one `score`, or every later frame["score"] returns a DataFrame.
        frame = self._frame(selftext="hi", score=0, points=99, created_utc=1700000000)
        out = normalize_columns(frame, "score=points", verbose=False)
        assert list(out.columns).count("score") == 1
        assert out["score"].iloc[0] == 99

    def test_missing_required_column_raises_with_guidance(self):
        frame = self._frame(selftext="hello", score=3)
        with pytest.raises(KeyError, match="created_utc"):
            normalize_columns(frame, verbose=False)

    def test_explicit_map_to_absent_source_raises(self):
        frame = self._frame(selftext="hello", score=3, created_utc=1)
        with pytest.raises(KeyError, match="not in"):
            normalize_columns(frame, "selftext=nope", verbose=False)

    def test_title_is_optional(self):
        frame = self._frame(selftext="hello", score=3, created_utc=1)
        assert "title" not in normalize_columns(frame, verbose=False).columns

    def test_does_not_mutate_the_input(self):
        frame = self._frame(body="hello", score=3, creation_date=1)
        normalize_columns(frame, verbose=False)
        assert "body" in frame.columns


class TestDescribeSchema:
    def test_reports_every_canonical_column(self):
        frame = pd.DataFrame({"selftext": ["a"], "score": [2], "created_utc": [1]})
        text = describe_schema(frame)
        for column in ("selftext", "score", "created_utc", "title"):
            assert column in text

    def test_marks_absent_columns(self):
        frame = pd.DataFrame({"selftext": ["a"], "score": [2], "created_utc": [1]})
        assert "(absent)" in describe_schema(frame)

    def test_reports_numeric_range(self):
        frame = pd.DataFrame({"selftext": ["a", "b"], "score": [-3, 40],
                              "created_utc": [1, 2]})
        assert "[-3, 40]" in describe_schema(frame)
