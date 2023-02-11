import logging
import random

import pytest

from min_hash import MinHasher, compare


def test_same():
    # Randomly generate sets of numbers and comfirm it's a perfect match with itself.
    num_hashes = 10
    hasher = MinHasher(num_hashes)
    for i in range(20):
        values = set(random.randint(0, 1000) for _ in range(100))
        assert compare(hasher(values), hasher(values)) == 1.0


def test_different():
    num_hashes = 2

    # Generate two lists with a 1/3 overlap
    values_a = [1, 2]
    values_b = [2, 3]

    # Record the mean of the similarity.
    sum_compare = 0.0
    NUM_COMPARISONS = 5000
    random.seed(4)
    for i in range(NUM_COMPARISONS):
        min_hasher = MinHasher(num_hashes, seed=i)

        min_hash_a = min_hasher(values_a)
        min_hash_b = min_hasher(values_b)
        compare_result = compare(min_hash_a, min_hash_b)
        sum_compare += compare_result

    # The mean should be close to 1/3
    target = len(set(values_a).intersection(values_b)) / len(set(values_a).union(values_b))
    EPS = 0.05
    mean_compare = sum_compare / NUM_COMPARISONS
    logging.warning("%s, %s", mean_compare, target)
    assert target - EPS < mean_compare
    assert mean_compare < target + EPS


def test_seed_mismatch():
    num_hashes = 10

    values_a = ["a", "b", "c"]
    values_b = ["b", "c", "d"]

    hasher1 = MinHasher(num_hashes, seed=1)
    hasher2 = MinHasher(num_hashes, seed=2)

    assert pytest.raises(ValueError, compare, hasher1(values_a), hasher2(values_b))
