"""
Vocabulary mappings for contiguous integer user and item IDs.
Strict rule: Vocabularies must only be fit on training interactions.
Unseen IDs in val/test are assigned to OOV (-1 or special OOV token).
"""

import json
from collections.abc import Iterable


class IdVocabulary:
    """Contiguous integer mapping for categorical IDs (e.g. user_id, movie_id)."""

    def __init__(self, name: str, oov_index: int = -1) -> None:
        self.name = name
        self.oov_index = oov_index
        self.id_to_index: dict[int, int] = {}
        self.index_to_id: dict[int, int] = {}

    def fit(self, ids: Iterable[int]) -> "IdVocabulary":
        """Fit vocabulary using only IDs observed in the training split."""
        unique_ids = sorted(set(ids))
        self.id_to_index = {raw_id: idx for idx, raw_id in enumerate(unique_ids)}
        self.index_to_id = {idx: raw_id for raw_id, idx in self.id_to_index.items()}
        return self

    def transform(self, raw_id: int) -> int:
        """Transform a raw ID to contiguous index; returns oov_index if unknown."""
        return self.id_to_index.get(raw_id, self.oov_index)

    def inverse_transform(self, index: int) -> int | None:
        """Transform contiguous index back to raw ID."""
        return self.index_to_id.get(index, None)

    def __len__(self) -> int:
        return len(self.id_to_index)

    def is_known(self, raw_id: int) -> bool:
        """Check if an ID was present in the training vocabulary."""
        return raw_id in self.id_to_index

    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON."""
        data = {
            "name": self.name,
            "oov_index": self.oov_index,
            "id_to_index": {str(k): v for k, v in self.id_to_index.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "IdVocabulary":
        """Load vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(name=data["name"], oov_index=data.get("oov_index", -1))
        vocab.id_to_index = {int(k): v for k, v in data["id_to_index"].items()}
        vocab.index_to_id = {v: int(k) for k, v in data["id_to_index"].items()}
        return vocab
