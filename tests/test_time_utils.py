"""Timestamp conversion and temporal feature engineering."""

import numpy as np
import pandas as pd
import pytest

from training.config import TEMPORAL_FEATURES
from utils.time_utils import (
    add_temporal_features,
    convert_to_datetime,
    hourly_distribution,
    late_night_share,
    temporal_feature_matrix,
)


class TestConvertToDatetime:
    def test_converts_epoch_seconds(self):
        out = convert_to_datetime(pd.DataFrame({"t": [1700000000]}), "t")
        assert out["t"].iloc[0].year == 2023

    def test_detects_millisecond_timestamps(self):
        # Treating ms as s lands you in the year 55000.
        out = convert_to_datetime(pd.DataFrame({"t": [1700000000000] * 5}), "t")
        assert out["t"].iloc[0].year == 2023

    def test_parses_iso_strings(self):
        out = convert_to_datetime(pd.DataFrame({"t": ["2023-05-01 10:00:00"]}), "t")
        assert out["t"].iloc[0].month == 5

    def test_already_datetime_is_passed_through(self):
        frame = pd.DataFrame({"t": pd.to_datetime(["2023-05-01"])})
        assert convert_to_datetime(frame, "t")["t"].iloc[0].year == 2023

    def test_missing_column_raises_keyerror(self):
        with pytest.raises(KeyError, match="nope"):
            convert_to_datetime(pd.DataFrame({"t": [1]}), "nope")

    def test_does_not_mutate_the_input(self):
        frame = pd.DataFrame({"t": [1700000000]})
        convert_to_datetime(frame, "t")
        assert frame["t"].dtype == np.int64


class TestTemporalFeatures:
    def test_adds_every_configured_feature(self):
        out = add_temporal_features(pd.DataFrame({"created_utc": [1700000000]}))
        for name in TEMPORAL_FEATURES:
            assert name in out.columns

    def test_cyclical_encoding_wraps_around(self):
        # 23:30 and 00:30 are half an hour apart; their encodings should be
        # close, even though the raw hour integers are 23 apart.
        base = pd.Timestamp("2023-06-01 23:30").timestamp()
        frame = pd.DataFrame({"created_utc": [base, base + 3600]})
        out = add_temporal_features(frame)
        distance = np.hypot(
            out["hour_sin"].iloc[1] - out["hour_sin"].iloc[0],
            out["hour_cos"].iloc[1] - out["hour_cos"].iloc[0],
        )
        # Compare against hours that really are far apart (00 vs 12).
        far = pd.DataFrame({"created_utc": [
            pd.Timestamp("2023-06-01 00:00").timestamp(),
            pd.Timestamp("2023-06-01 12:00").timestamp(),
        ]})
        far_out = add_temporal_features(far)
        far_distance = np.hypot(
            far_out["hour_sin"].iloc[1] - far_out["hour_sin"].iloc[0],
            far_out["hour_cos"].iloc[1] - far_out["hour_cos"].iloc[0],
        )
        assert distance < far_distance

    def test_cyclical_features_lie_on_the_unit_circle(self):
        out = add_temporal_features(pd.DataFrame({
            "created_utc": [1700000000 + i * 3600 for i in range(24)]
        }))
        radius = out["hour_sin"] ** 2 + out["hour_cos"] ** 2
        assert np.allclose(radius, 1.0)

    def test_late_night_flag(self):
        hours = [pd.Timestamp(f"2023-06-01 {h:02d}:00").timestamp() for h in (2, 4, 5, 13)]
        out = add_temporal_features(pd.DataFrame({"created_utc": hours}))
        assert out["is_late_night"].tolist() == [1.0, 1.0, 0.0, 0.0]

    def test_weekend_flag(self):
        # 2023-06-03 is a Saturday, 2023-06-05 a Monday.
        stamps = [pd.Timestamp("2023-06-03").timestamp(), pd.Timestamp("2023-06-05").timestamp()]
        out = add_temporal_features(pd.DataFrame({"created_utc": stamps}))
        assert out["is_weekend"].tolist() == [1.0, 0.0]

    def test_missing_timestamps_do_not_produce_nan_features(self):
        frame = pd.DataFrame({"created_utc": [1700000000, None, 1700100000]})
        matrix = temporal_feature_matrix(add_temporal_features(frame), TEMPORAL_FEATURES)
        assert not np.isnan(matrix).any()

    def test_all_timestamps_unparseable_raises(self):
        with pytest.raises(ValueError, match="no parseable timestamps"):
            add_temporal_features(pd.DataFrame({"created_utc": ["x", "y"]}))


class TestFeatureMatrix:
    def test_shape_and_dtype(self, clean_frame):
        out = add_temporal_features(clean_frame)
        matrix = temporal_feature_matrix(out, TEMPORAL_FEATURES)
        assert matrix.shape == (len(clean_frame), len(TEMPORAL_FEATURES))
        assert matrix.dtype == np.float32

    def test_missing_feature_raises_with_guidance(self, clean_frame):
        with pytest.raises(KeyError, match="add_temporal_features"):
            temporal_feature_matrix(clean_frame, ["hour_sin"])

    def test_column_order_is_respected(self):
        out = add_temporal_features(pd.DataFrame({"created_utc": [1700000000]}))
        forward = temporal_feature_matrix(out, ["hour_sin", "hour_cos"])
        reverse = temporal_feature_matrix(out, ["hour_cos", "hour_sin"])
        assert forward[0, 0] == reverse[0, 1]


class TestDistributions:
    def test_hourly_distribution_covers_all_24_hours(self, clean_frame):
        dist = hourly_distribution(clean_frame)
        assert len(dist) == 24
        assert np.isclose(dist.sum(), 1.0)

    def test_late_night_share_is_a_fraction(self, clean_frame):
        assert 0.0 <= late_night_share(clean_frame) <= 1.0

    def test_late_night_share_on_empty_frame(self):
        empty = pd.DataFrame({"created_utc": pd.Series([], dtype="float64")})
        assert late_night_share(empty) == 0.0
