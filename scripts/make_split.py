"""Writes splits/val_rows.json: the train rows held out for validation.

Run once; the file is committed so every machine uses the same split. It is
a stratified 10% sample of train (seed 77), so every intent appears in
validation in the same proportion as in train.
"""

import json

from tunebench.data import VAL_ROWS, choose_val_rows, read_split

rows = choose_val_rows(read_split("train").labels)
VAL_ROWS.parent.mkdir(exist_ok=True)
VAL_ROWS.write_text(json.dumps(rows) + "\n")
print(f"{len(rows)} validation rows written to {VAL_ROWS}")
