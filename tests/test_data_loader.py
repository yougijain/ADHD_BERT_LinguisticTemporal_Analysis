"""Dataset construction and train/val splitting.

Replaces the old data/test_data_loader.py, which built its encodings from
ragged nested lists ([101, 2009, 2001, 1037, 3867, 102] next to
[101, 1045, 2293, 2023, 102] -- 6 tokens and 5) and so crashed inside
torch.tensor before reaching a single assertion. It also had no assertions; it
printed batches and relied on a human reading the output.
"""

import numpy as np
import pytest
import torch

from data.data_loader import ADHDTextDataset, build_dataloaders, split_indices


@pytest.fixture
def encodings():
    return {
        "input_ids": torch.randint(0, 100, (10, 8)),
        "attention_mask": torch.ones(10, 8, dtype=torch.long),
    }


class TestADHDTextDataset:
    def test_length(self, encodings):
        assert len(ADHDTextDataset(encodings)) == 10

    def test_item_without_labels_has_no_label_key(self, encodings):
        assert "labels" not in ADHDTextDataset(encodings)[0]

    def test_item_with_labels(self, encodings):
        item = ADHDTextDataset(encodings, labels=list(range(10)))[3]
        assert item["labels"].item() == 3
        assert item["labels"].dtype == torch.long

    def test_temporal_features_are_carried_through(self, encodings):
        features = np.random.randn(10, 8).astype("float32")
        item = ADHDTextDataset(encodings, labels=[0] * 10, temporal_features=features)[0]
        assert item["temporal_features"].shape == (8,)
        assert item["temporal_features"].dtype == torch.float32

    def test_ragged_encodings_raise_an_actionable_error(self):
        # The exact input the old test file used.
        ragged = {"input_ids": [[101, 2009, 2001, 1037, 3867, 102], [101, 1045, 2293, 2023, 102]]}
        with pytest.raises(ValueError, match="padding"):
            ADHDTextDataset(ragged)

    def test_label_length_mismatch_raises(self, encodings):
        with pytest.raises(ValueError, match="same length"):
            ADHDTextDataset(encodings, labels=[0, 1])

    def test_temporal_length_mismatch_raises(self, encodings):
        with pytest.raises(ValueError, match="align"):
            ADHDTextDataset(encodings, temporal_features=np.zeros((3, 8)))

    def test_missing_input_ids_raises(self):
        with pytest.raises(KeyError, match="input_ids"):
            ADHDTextDataset({"attention_mask": torch.ones(2, 4)})

    def test_num_temporal_features_property(self, encodings):
        assert ADHDTextDataset(encodings).num_temporal_features == 0
        with_features = ADHDTextDataset(encodings, temporal_features=np.zeros((10, 5)))
        assert with_features.num_temporal_features == 5

    def test_1d_temporal_features_are_promoted(self, encodings):
        dataset = ADHDTextDataset(encodings, temporal_features=np.arange(10))
        assert dataset.num_temporal_features == 1

    def test_getitem_does_not_alias_the_source(self, encodings):
        dataset = ADHDTextDataset(encodings)
        item = dataset[0]
        item["input_ids"][0] = 999
        assert dataset.encodings["input_ids"][0, 0] != 999


class TestSplitIndices:
    def test_temporal_split_preserves_order(self):
        train, val = split_indices(100, 0.2, "temporal")
        assert train.tolist() == list(range(80))
        assert val.tolist() == list(range(80, 100))

    def test_random_split_shuffles_but_partitions(self):
        train, val = split_indices(100, 0.2, "random", seed=1)
        assert sorted(train.tolist() + val.tolist()) == list(range(100))
        assert train.tolist() != list(range(80))

    def test_random_split_is_deterministic_for_a_seed(self):
        assert split_indices(50, 0.2, "random", seed=3)[1].tolist() == \
               split_indices(50, 0.2, "random", seed=3)[1].tolist()

    def test_splits_never_overlap(self):
        train, val = split_indices(37, 0.3, "random", seed=5)
        assert set(train.tolist()).isdisjoint(val.tolist())

    def test_validation_set_is_never_empty(self):
        # A tiny dataset with a small val_split used to round down to zero
        # validation rows, making every metric silently meaningless.
        for n in (2, 3, 5, 9):
            train, val = split_indices(n, 0.1, "temporal")
            assert len(val) >= 1 and len(train) >= 1

    def test_empty_dataset(self):
        train, val = split_indices(0)
        assert len(train) == 0 and len(val) == 0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown split strategy"):
            split_indices(10, 0.2, "sideways")


class TestBuildDataloaders:
    def test_batches_contain_every_expected_key(self, encodings):
        train, _ = build_dataloaders(
            encodings, labels=np.array([0, 1] * 5),
            temporal_features=np.random.randn(10, 8).astype("float32"),
            batch_size=2,
        )
        batch = next(iter(train))
        assert set(batch) == {"input_ids", "attention_mask", "temporal_features", "labels"}

    def test_sizes_match_the_requested_split(self, encodings):
        train, val = build_dataloaders(encodings, np.array([0, 1] * 5), batch_size=2,
                                       val_split=0.2)
        assert len(train.dataset) == 8 and len(val.dataset) == 2

    def test_works_without_temporal_features(self, encodings):
        train, _ = build_dataloaders(encodings, np.array([0, 1] * 5), batch_size=2)
        assert "temporal_features" not in next(iter(train))
