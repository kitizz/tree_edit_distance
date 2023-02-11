from collections.abc import Iterable, Hashable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar
import numpy as np

T = TypeVar("T")
HashFunction = Callable[[T], int]


@dataclass
class MinHash:
    signature: tuple[int, ...]
    _hash_seed: int


class MinHasher(Generic[T]):
    num_hashes: int
    hash_func: HashFunction[T]
    hash_parameters: list[tuple[int, int]]
    seed: int

    # For binning the hash values.
    LARGE_PRIME = 1073741827

    def __init__(self, num_hashes: int, hash_func: HashFunction[T] = hash, seed: int = 1):
        self.num_hashes = num_hashes
        self.hash_func = hash_func
        self.seed = seed

        gen = np.random.default_rng(seed)
        parameters = gen.integers(0, 0xFFFFFFFF, size=(num_hashes, 2), dtype=np.uint32)
        self.hash_parameters = [(int(a), int(b)) for a, b in parameters]

    def __call__(self, values: Iterable[T]) -> MinHash:
        # Hash each value and keep the num_hashes smallest of them
        hashes = [self.LARGE_PRIME] * self.num_hashes
        for value in values:
            for i, params in enumerate(self.hash_parameters):
                hashed = self._hash(value, params)
                hashes[i] = min(hashes[i], hashed)

        return MinHash(tuple(hashes), self.seed)

    def _hash(self, value: T, hash_params: tuple[int, int]) -> int:
        a, b = hash_params
        return (self._hash_int(self.hash_func(value) * a) + b) % self.LARGE_PRIME

    def _hash_int(self, value: int) -> int:
        """Map value to a random value 32-bit int with uniform distribution.

        This helps avoid some bias that can occur when inputs are small integers.
        """
        # See: https://stackoverflow.com/a/12996028/1512137
        value = ((value >> 16) ^ value) * 0x45D9F3B
        value = ((value >> 16) ^ value) * 0x45D9F3B
        value = (value >> 16) ^ value
        return value


def compare(a: MinHash, b: MinHash) -> float:
    """Compare two MinHash signatures and return the fraction of hashes that match.

    This approximates the Jaccard similarity of the two sets used to create the MinHashes.
    """
    if len(a.signature) != len(b.signature):
        raise ValueError("Signatures must be the same length")
    if a._hash_seed != b._hash_seed:
        raise ValueError("Signatures must be generated with the same seed")

    # Count the number of hashes that are in both signatures
    num_matches = sum(a == b for a, b in zip(a.signature, b.signature))
    return num_matches / len(a.signature)
