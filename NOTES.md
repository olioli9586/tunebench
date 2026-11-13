# Notes

A running log: what was done, what was decided and why, what is next.

## 2026-11-12: the whole first version in one day

The plan was about three weeks of evenings on Kaggle's free GPUs. The laptop
(Apple M5 Pro, 24 GB) trains ModernBERT-base in about 10 minutes per seed on
its GPU (PyTorch MPS) and the Qwen3 LoRA in about 18, so everything ran
locally in one day.

Done:
- Data: the two Banking77 parquet files, pinned by sha256. Validation is a
  stratified 10% of train (seed 77), rows committed in splits/val_rows.json.
  Test (3,080, 40 per intent) is used once per model, after training.
- Baseline: TF-IDF (word 1-2 + char 2-5 grams) + logistic regression, C=100
  picked on validation (the grid was widened to 1000 after 100 came out on
  the edge; 100 stayed best).
- Claude Haiku 5.5 zero-shot and 10-shot (TF-IDF nearest training messages),
  effort high, structured output limited to the 77 names, hard cap per run.
  Spent $1.57 in total, including two 40-message trials.
- ModernBERT-base full fine-tune, 3 seeds (93.4 ± 0.5%). Qwen3-0.6B LoRA
  rank sweep (8, 16, 32) on seed 0, rank 16 chosen on validation, then 3
  seeds (93.3 ± 0.1%). MLflow in mlflow.db (local only: it holds absolute paths).
- ONNX export (torch.onnx dynamo exporter) of the best-validation seed
  (seed 2), dynamic per-channel int8, FastAPI server without PyTorch,
  Dockerfile, latency scripts, report.
- CI: ruff, pytest without PyTorch, Docker build plus a smoke test that
  serves a tiny random model checked in under tests/fixtures/tiny.

Decided, and why:
- Qwen3-0.6B, not the newer Qwen3.5-0.8B: Qwen3.5 is a vision-language model
  whose layers are mostly Gated DeltaNet linear attention; the fast kernels
  for those need CUDA/Triton, so on an Apple GPU it would fall back to slow
  reference code. Qwen3 is a plain transformer with a ready
  sequence-classification class.
- A classification head on the LLM, not generated label text: one forward
  pass, always one of the 77 intents, scored exactly like the encoder.
- MLflow instead of W&B: local, no account needed.
- Learning rates are standard values (5e-5 full fine-tune, 2e-4 LoRA), not
  tuned; C, the LoRA rank, the int8 scheme and the served seed are chosen on
  validation.
- int8 scheme: per-channel vs per-tensor scales, scored on validation with
  scripts/quant_variants.py. On seed 0 they tied; on seed 2, the served
  model, per-channel lost 0.4 points against float32 and per-tensor 0.6, so
  the export uses per-channel (test: 93.6% vs float32 93.8%).
- Latency in batches was dropped from the benchmark: a batch pads to its
  longest message, so on short messages it does 3-4x the work of single
  messages and reads slower per message, which says nothing useful about
  serving one request at a time.
- LoRA early stopping went from patience 1 to 2. With 1, the first rank-16
  run stopped after a single dip in validation macro-F1 at epoch 2 (0.881 ->
  0.877; test 88.4%), while ranks 8 and 32 kept improving to epochs 4-5.
  Validation F1 moves about a point between epochs, so one dip is noise.
  Rank 16 was rerun with patience 2; ranks 8 and 32 would have run exactly
  the same epochs under either setting.
- Latency is measured only on an idle machine. The first benchmark ran while
  the GPU was training and came out up to 3x slower; the GPU job keeps the
  CPU busy feeding it.
- Throughput for the cost estimate is 2 clients against 1 thread per request
  (about two cores of work). The first version used 8 clients against 2
  threads each, which kept far more than 2 cores busy and flattered a 2-vCPU
  cost by about 3x.
- Docker Desktop does not start on this Mac, so the image is built and
  smoke-tested in CI only; latency is measured with the same server code run
  natively.

A fresh-context review found, and these were fixed:
- claude.py: messages.parse raises on an answer cut off mid-JSON, which would
  have ended the run and dropped the requests in flight. Now the same schema
  goes through messages.create (a test checks it matches what parse sends),
  a bad answer counts as wrong with its tokens counted, and a failed request
  lets the others finish and be logged. Neither reported run hit this (0
  unanswered). The per-request ceiling now allows 4,000 prompt tokens.
- train.py: on a float16 GPU (the notebook's T4) the new head stayed in
  float16 and the gradient scaler would refuse it; trainable weights are now
  float32 there. The LoRA merge now happens in float32 (into bfloat16 it
  rounded much of the update away). The best epoch is read from the step
  Trainer kept. Default names include the rank, results files are not
  overwritten silently, smoke runs leave no trace.
- Claims in the README that the files didn't support (tuned learning rates,
  250 extra tokens, a per-channel test with no file, "ties", "billed").
Known and left: on bfloat16 the Qwen head trains without a float32 copy, so
its smallest late updates round away. Changing it means retraining every
LoRA run; it is the next experiment.

Findings worth remembering:
- Claude 10-shot (94.3%) and ModernBERT (seed 0, 93.6%) are not
  significantly different (McNemar p = 0.11). Served as int8, ModernBERT
  costs about $0.000075 per 1,000 requests on a 2-vCPU instance against
  $0.258 for Claude (over 3,000x) and answers in about 4 ms against 0.89 s.
- Zero-shot Claude (81.9%) is far below even the TF-IDF baseline (91.1%):
  the intent names alone don't say where Banking77 draws its lines
  (card_arrival vs card_delivery_estimate). Ten labeled neighbors fix most
  of that (+12 points), and Claude then thinks less (47 output tokens
  instead of 94), so 10-shot costs only 4% more.
- Both models miss the same 111 test messages, 80 of them with the same
  wrong answer. Reading 14, four gold labels look wrong ("my card keeps
  getting declined" labeled declined_transfer) and most of the rest sit
  between two near-duplicate intents.

Next:
- Hugging Face: model repo and the demo Space need the owner's account
  (`.venv/bin/hf auth login`, then `python scripts/publish.py modernbert-seed2`).
  Then check the Space's int8 answers on x86 against the local ones.
- EvalBoard "custom HTTP endpoint" model type (olioli9586/evalboard#4), so
  the served model and Claude run through the same eval.
