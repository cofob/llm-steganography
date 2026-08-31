import numpy as np
import pytest

from llm_steganography.partition import TokenPartitioner

KEY = bytes(range(32))


def test_golden_token_labels() -> None:
    partitioner = TokenPartitioner(KEY)
    token_ids = (0, 1, 2, 42, 999, 151665)
    assert [partitioner.label_token(token, []) for token in token_ids] == [0, 0, 1, 0, 1, 1]
    assert [partitioner.label_token(token, [10, 20, 30]) for token in token_ids] == [
        0,
        1,
        1,
        1,
        0,
        1,
    ]


def test_vector_and_single_token_paths_match() -> None:
    partitioner = TokenPartitioner(KEY)
    history = [10, 20, 30]
    labels = partitioner.labels_for_vocab(history, 1000)
    expected = np.asarray(
        [partitioner.label_token(token, history) for token in range(1000)],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(labels, expected)


def test_partition_is_approximately_balanced() -> None:
    labels = TokenPartitioner(KEY).labels_for_vocab([10, 20, 30], 1000)
    assert 450 <= int(labels.sum()) <= 550


def test_different_key_changes_partition() -> None:
    history = [1, 2, 3]
    first = TokenPartitioner(KEY).labels_for_vocab(history, 1000)
    second = TokenPartitioner(bytes(reversed(KEY))).labels_for_vocab(history, 1000)
    assert int(np.count_nonzero(first != second)) > 400


@pytest.mark.parametrize("groups", range(2, 11))
def test_partition_supports_multiple_group_counts(groups: int) -> None:
    partitioner = TokenPartitioner(KEY, groups)
    history = [10, 20, 30]
    labels = partitioner.labels_for_vocab(history, 10_000)
    assert int(labels.min()) == 0
    assert int(labels.max()) == groups - 1
    expected = len(labels) / groups
    counts = np.bincount(labels, minlength=groups)
    assert all(abs(int(count) - expected) < expected * 0.15 for count in counts)
    assert all(
        partitioner.label_token(token_id, history) == int(labels[token_id])
        for token_id in (0, 1, 42, 999, 9_999)
    )


@pytest.mark.parametrize("groups", [1, 11])
def test_partition_rejects_invalid_group_count(groups: int) -> None:
    with pytest.raises(ValueError, match="between 2 and 10"):
        TokenPartitioner(KEY, groups)
