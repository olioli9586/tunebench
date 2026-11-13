"""Approaches 2 and 3: fine-tune a pretrained model to pick one of the 77 intents.

    python -m tunebench.train --model modernbert --seed 0
    python -m tunebench.train --model qwen3-lora --seed 0 --rank 16

modernbert: ModernBERT-base (149M parameters), every weight trained.
qwen3-lora: Qwen3-0.6B with a classification head on its last token. The base
weights stay frozen; LoRA adapters on every linear layer and the new head are
trained. A head is simpler to score than generated label text: one forward
pass, one of 77 answers, never an invalid one.

Both share one loop (Hugging Face Trainer): validation after every epoch,
early stopping on validation macro-F1, the best epoch kept, then a single
pass over the test set. Each run is tracked in MLflow (sqlite:///mlflow.db,
`mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse) and its test
predictions written to results/. The model is saved to artifacts/<name>/ for
the ONNX export.
"""

import argparse
import os
import time
from dataclasses import dataclass, replace

import numpy as np

from .data import ROOT, Split, load
from .metrics import scores
from .results import RESULTS_DIR, record


@dataclass(frozen=True)
class Recipe:
    checkpoint: str
    revision: str
    lora: bool
    learning_rate: float
    epochs: int
    batch_size: int
    patience: int
    rank: int = 16


# Revisions pinned so a rerun downloads the same weights.
RECIPES = {
    "modernbert": Recipe(
        checkpoint="answerdotai/ModernBERT-base",
        revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
        lora=False,
        learning_rate=5e-5,
        epochs=10,
        batch_size=32,
        patience=2,
    ),
    "qwen3-lora": Recipe(
        checkpoint="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        lora=True,
        learning_rate=2e-4,
        epochs=5,
        batch_size=16,
        patience=2,
    ),
}
MAX_LENGTH = 128  # the longest Banking77 message is well under this; batches pad to their longest
CHECKPOINTS = ROOT / "checkpoints"
ARTIFACTS = ROOT / "artifacts"


class Encoded:
    """A split tokenized once, served to Trainer as dicts of token ids."""

    def __init__(self, split: Split, tokenizer):
        self.encodings = tokenizer(split.texts, truncation=True, max_length=MAX_LENGTH)
        self.labels = split.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, object]:
        return {key: values[i] for key, values in self.encodings.items()} | {"labels": self.labels[i]}


def build(recipe: Recipe, names: list[str]):
    """Tokenizer and model for a recipe. For LoRA, only adapters and the head are trainable."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(recipe.checkpoint, revision=recipe.revision)
    options: dict[str, object] = {}
    if recipe.lora:
        # Frozen base weights in 16 bits halve memory; the adapters train in float32.
        options["dtype"] = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision(), torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        recipe.checkpoint,
        revision=recipe.revision,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={name: i for i, name in enumerate(names)},
        **options,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The head reads the last real token, found through the pad id.
    model.config.pad_token_id = tokenizer.pad_token_id
    if recipe.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=recipe.rank,
            lora_alpha=2 * recipe.rank,
            lora_dropout=0.05,
            target_modules="all-linear",
        )
        model = get_peft_model(model, config)
        if precision() == "fp16":
            from peft import cast_mixed_precision_params

            # float16 training needs float32 trainable weights (the gradient scaler
            # refuses float16 ones); the new head would otherwise stay in float16.
            # On bfloat16 the head stays in bfloat16, as in the reported runs.
            cast_mixed_precision_params(model, torch.float16)
    return tokenizer, model


def precision() -> str:
    """Mixed precision for this device: bfloat16 on Apple GPUs and recent NVIDIA ones,
    float16 on older NVIDIA cards without bfloat16 (Kaggle's and Colab's T4), none on a CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "bf16"
    if torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def parameter_counts(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def run(model_key: str, seed: int, recipe: Recipe, name: str, max_train: int | None = None) -> dict:
    import contextlib

    import mlflow
    from transformers import (
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    data = load()
    train = data.train
    if max_train:  # smoke runs only
        train = Split(train.texts[:max_train], train.labels[:max_train])
    tokenizer, model = build(recipe, data.names)

    def compute_metrics(prediction) -> dict[str, float]:
        return scores(list(prediction.label_ids), np.argmax(prediction.predictions, axis=-1).tolist())

    args = TrainingArguments(
        output_dir=str(CHECKPOINTS / name),
        learning_rate=recipe.learning_rate,
        per_device_train_batch_size=recipe.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=recipe.epochs,
        warmup_steps=0.1,
        weight_decay=0.01,
        bf16=precision() == "bf16",
        fp16=precision() == "fp16",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        data_seed=seed,
        report_to=["none"] if max_train else ["mlflow"],
        run_name=name,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Encoded(train, tokenizer),
        eval_dataset=Encoded(data.val, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=recipe.patience)],
    )

    tracked = not max_train  # smoke runs leave no trace in MLflow or artifacts/
    if tracked:
        os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("tunebench")
    # A run opened here stays open after Trainer finishes, so the test scores join it.
    with mlflow.start_run(run_name=name) if tracked else contextlib.nullcontext():
        if tracked:
            mlflow.log_params({"model": model_key, "checkpoint": recipe.checkpoint, "seed": seed})
        started = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - started
        val = trainer.evaluate()
        test = trainer.predict(Encoded(data.test, tokenizer))
        predicted = np.argmax(test.predictions, axis=-1).tolist()
        best_step = trainer.state.best_global_step
        best_epoch = next(
            (e["epoch"] for e in trainer.state.log_history if "eval_macro_f1" in e and e["step"] == best_step),
            None,
        )
        row = record(
            name,
            "lora" if recipe.lora else "encoder",
            data.test.labels,
            predicted,
            # Smoke runs (a slice of train) stay out of results/.
            **({"results_dir": CHECKPOINTS / name} if max_train else {}),
            model=recipe.checkpoint,
            revision=recipe.revision,
            seed=seed,
            rank=recipe.rank if recipe.lora else None,
            learning_rate=recipe.learning_rate,
            best_epoch=best_epoch,
            val_accuracy=val["eval_accuracy"],
            val_macro_f1=val["eval_macro_f1"],
            train_seconds=round(train_seconds),
            device=str(trainer.args.device),
            **parameter_counts(model),
        )
        if tracked:
            mlflow.log_metrics(
                {"test_accuracy": row["accuracy"], "test_macro_f1": row["macro_f1"], "train_seconds": train_seconds}
            )

    if not tracked:
        return row
    if recipe.lora:
        # Merge in float32: adding the small LoRA update to bfloat16 weights
        # would round much of it away, and the saved model would not be the one scored.
        model = model.float().merge_and_unload()
    out = ARTIFACTS / name
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(RECIPES), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, help="LoRA rank (alpha is twice the rank)")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train", type=int, help="train on the first N examples (smoke runs)")
    parser.add_argument(
        "--name", help="results/<name>.json; default <model>-seed<seed> (LoRA: <model>-r<rank>-seed<seed>)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing results file of the same name")
    a = parser.parse_args()
    recipe = RECIPES[a.model]
    changes = {
        k: v for k, v in {"rank": a.rank, "learning_rate": a.learning_rate, "epochs": a.epochs}.items() if v is not None
    }
    recipe = replace(recipe, **changes)
    name = a.name or (f"{a.model}-r{recipe.rank}-seed{a.seed}" if recipe.lora else f"{a.model}-seed{a.seed}")
    if not a.max_train and (RESULTS_DIR / f"{name}.json").exists() and not a.overwrite:
        parser.error(f"results/{name}.json exists; pass --overwrite to replace it, or --name.")
    row = run(a.model, a.seed, recipe, name, a.max_train)
    print(
        f"{name}: test accuracy {row['accuracy']:.4f}, macro-F1 {row['macro_f1']:.4f}, "
        f"best epoch {row['best_epoch']}, {row['train_seconds']} s on {row['device']}"
    )


if __name__ == "__main__":
    main()
