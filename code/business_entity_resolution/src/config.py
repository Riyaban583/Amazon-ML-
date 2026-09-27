import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Directory paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "dataset"
    TRAIN_DIR: Path = DATA_DIR / "train"
    TEST_DIR: Path = DATA_DIR / "test"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    MODEL_DIR: Path = BASE_DIR / "models"

    # Blocking & Candidate Generation parameters (Riya's module)
    TOP_CANDIDATES_PER_S1: int = 25  # keeps candidate set small & high precision
    MIN_TOKEN_LEN_FOR_INDEX: int = 3
    MAX_TOKEN_INDEX_FREQ: float = 0.05  # filter overly common generic words
    JACCARD_PRUNING_THRESHOLD: float = 0.08  # discard pairs with almost zero lexical overlap

    # Feature Engineering (Khushi's module)
    USE_TFIDF: bool = True
    CHAR_NGRAM_RANGE: tuple = (3, 3)
    TFIDF_MAX_FEATURES: int = 512

    # ML Model & Evaluation (Akshat's module)
    SEED: int = 42
    N_FOLDS: int = 5
    MATCH_PROB_THRESHOLD: float = 0.45  # Secondary threshold for multi-match
    SINGLETON_PROB_THRESHOLD: float = 0.62  # High bar required to avoid false merge on singleton
    MAX_MATCHES_PER_S1: int = 10  # Plausible maximum matches per S1

    def ensure_directories(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
