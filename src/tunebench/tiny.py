"""A tiny, randomly initialized ModernBERT classifier with the real model's interface.

Tests and CI use it to exercise the export and serving code end to end in
seconds, without downloading or committing the real model. Its predictions
are meaningless; its inputs, outputs and files are the same shape as the
real thing. scripts/make_fixture.py exports one into tests/fixtures/tiny/.
"""

from collections import Counter
from pathlib import Path

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def make_tiny_model(out_dir: Path, texts: list[str], names: list[str], vocab_size: int = 2000) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import ModernBertConfig, ModernBertForSequenceClassification, PreTrainedTokenizerFast

    words = Counter(w for t in texts for w in t.lower().split())
    vocab = {token: i for i, token in enumerate(SPECIALS)}
    for word, _ in words.most_common(vocab_size - len(SPECIALS)):
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        pad_token_id=vocab["[PAD]"],
        cls_token_id=vocab["[CLS]"],
        sep_token_id=vocab["[SEP]"],
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)},
    )
    model = ModernBertForSequenceClassification(config)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
