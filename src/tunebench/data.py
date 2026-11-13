"""Banking77: 13,083 messages to a bank's support team, each labeled with one of 77 intents.

The data is PolyAI/banking77 on Hugging Face (CC BY 4.0). Its two parquet
files are downloaded once into data/ and checked against the hashes below, so
every machine trains and tests on the same bytes. The official test split
(3,080 messages) is the held-out test set for every approach. Validation is
carved from train, stratified by intent with a fixed seed, and the chosen row
numbers are committed in splits/val_rows.json.
"""

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
VAL_ROWS = ROOT / "splits" / "val_rows.json"
LABELS = Path(__file__).with_name("labels.json")

PARQUET_URL = "https://huggingface.co/api/datasets/PolyAI/banking77/parquet/default/{split}/0.parquet"
SHA256 = {
    "train": "8dc89b87a188509c42cce5a086e8f7c1ff4a225bdd9ae3f23e6cdfa5806e363a",
    "test": "1f40f777c33f0edfa6c530ff84fe2878db6490838c12d22d036f626948905341",
}
VAL_FRACTION = 0.1
SPLIT_SEED = 77


@dataclass(frozen=True)
class Split:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class Banking77:
    train: Split
    val: Split
    test: Split
    names: list[str]


def label_names() -> list[str]:
    return json.loads(LABELS.read_text())


def download(split: str, data_dir: Path = DATA_DIR) -> Path:
    """The split's parquet file, fetched on first use and checked by hash."""
    path = data_dir / f"{split}.parquet"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PARQUET_URL.format(split=split), timeout=60) as res:
            body = res.read()
        path.write_bytes(body)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[split]:
        raise ValueError(
            f"{path} has sha256 {digest}, not the pinned {SHA256[split]}. "
            "Delete it to download again; if Hugging Face changed the file, the pin needs review."
        )
    return path


def read_split(split: str, data_dir: Path = DATA_DIR) -> Split:
    import pyarrow.parquet as pq

    table = pq.read_table(download(split, data_dir)).to_pydict()
    return Split(texts=table["text"], labels=table["label"])


def choose_val_rows(labels: list[int], fraction: float = VAL_FRACTION, seed: int = SPLIT_SEED) -> list[int]:
    """Train row numbers for validation: a stratified sample, the same for a given seed."""
    from sklearn.model_selection import train_test_split

    _, val = train_test_split(list(range(len(labels))), test_size=fraction, stratify=labels, random_state=seed)
    return sorted(val)


def carve(full: Split, val_rows: list[int]) -> tuple[Split, Split]:
    chosen = set(val_rows)
    if len(chosen) != len(val_rows) or not all(0 <= i < len(full) for i in chosen):
        raise ValueError("Validation rows must be distinct row numbers within the train split.")
    keep = [i for i in range(len(full)) if i not in chosen]
    pick = lambda rows: Split([full.texts[i] for i in rows], [full.labels[i] for i in rows])  # noqa: E731
    return pick(keep), pick(val_rows)


def load(data_dir: Path = DATA_DIR) -> Banking77:
    full = read_split("train", data_dir)
    train, val = carve(full, json.loads(VAL_ROWS.read_text()))
    return Banking77(train=train, val=val, test=read_split("test", data_dir), names=label_names())
