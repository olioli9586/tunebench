"""Builds RESULTS.md and docs/accuracy.png from the files in results/. No model runs.

    python scripts/report.py

Every number in RESULTS.md comes from results/*.json (one file per run, with
its test predictions) and results/serving/*.json (export and latency), so the
report can be rebuilt and checked from the repo alone.
"""

import json
import math
import re
import statistics
from pathlib import Path

from tunebench.data import ROOT, label_names, read_split
from tunebench.metrics import mcnemar, most_confused
from tunebench.results import load_all

SERVING = ROOT / "results" / "serving"
# A small cloud instance for the self-hosting cost estimate: AWS c7g.large
# (2 vCPU, 4 GB, Graviton3), on-demand in us-east-1, checked 2026-11-12
# (doit.com and holori.com price listings).
INSTANCE = "AWS c7g.large (2 vCPU)"
INSTANCE_USD_PER_HOUR = 0.0725


def family(name: str) -> str:
    return re.sub(r"-seed\d+$", "", name)


def label(fam: str) -> str:
    if fam == "tfidf-logreg":
        return "TF-IDF + logistic regression"
    if fam == "modernbert":
        return "ModernBERT-base, full fine-tune"
    if m := re.fullmatch(r"qwen3-lora-r(\d+)", fam):
        return f"Qwen3-0.6B + LoRA (rank {m[1]})"
    if m := re.fullmatch(r"claude-haiku-5-5-(\d+)shot-(\w+)", fam):
        shots = "zero-shot" if m[1] == "0" else f"{m[1]}-shot"
        return f"Claude Haiku 5.5, {shots} (effort {m[2]})"
    return fam


def dollars(x: float) -> str:
    """Two significant figures, however small: $0.000025, not $0.0000."""
    return f"${x:.2g}" if x >= 0.01 else f"${x:.{1 - math.floor(math.log10(x))}f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def spread(values: list[float]) -> str:
    if len(values) == 1:
        return pct(values[0])
    return f"{100 * statistics.mean(values):.1f} ± {100 * statistics.stdev(values):.1f}%"


def spread_f1(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    runs = load_all()
    names = label_names()
    truth = read_split("test").labels
    by_family: dict[str, list[dict]] = {}
    for run in runs:
        by_family.setdefault(family(run["name"]), []).append(run)

    # The LoRA rank in the main table is the one with the best seed-0 validation macro-F1.
    lora = {f: rs for f, rs in by_family.items() if f.startswith("qwen3-lora")}
    sweep = sorted(
        ((f, next(r for r in rs if r["seed"] == 0)) for f, rs in lora.items() if any(r["seed"] == 0 for r in rs)),
        key=lambda fr: fr[1]["rank"],
    )
    chosen_lora = max(sweep, key=lambda fr: fr[1]["val_macro_f1"])[0] if sweep else None
    main_families = [f for f in ("tfidf-logreg", "modernbert", chosen_lora) if f in by_family] + sorted(
        f for f in by_family if f.startswith("claude")
    )

    encoder_seeds = sorted(by_family.get("modernbert", []), key=lambda r: r["seed"])
    # The served model is the ModernBERT seed with the best validation macro-F1.
    best_seed = max(encoder_seeds, key=lambda r: r["val_macro_f1"], default=None)
    onnx = read(SERVING / f"onnx-{best_seed['name']}.json") if best_seed else None
    latency = read(SERVING / f"latency-{onnx['model']}.json") if onnx else None

    def torch_p50(fam: str) -> str:
        files = sorted(SERVING.glob(f"torch-latency-{fam}-seed*.json"))
        return f"{read(files[0])['latency_ms']['p50']:.0f} ms (PyTorch)" if files else "—"

    rows = []
    for fam in main_families:
        rs = by_family[fam]
        first = rs[0]
        if first["approach"] == "baseline":
            p50, cost = f"{first['latency_ms']['p50']:.1f} ms", "≈ $0"
        elif first["approach"] == "claude":
            p50 = f"{first['latency_ms']['p50'] / 1000:.2f} s (API)"
            cost = f"${first['cost_usd_per_1k']:.3f}"
        else:
            p50, cost = torch_p50(fam), "—"
        trained = f"{first['trainable_params'] / 1e6:.1f}M" if "trainable_params" in first else "—"
        rows.append(
            {
                "label": label(fam),
                "runs": len(rs),
                "acc": [r["accuracy"] for r in rs],
                "f1": [r["macro_f1"] for r in rs],
                "p50": p50,
                "cost": cost,
                "trained": trained,
            }
        )
    served = None
    if onnx and latency:
        rps = latency["int8"]["http"]["requests_per_second_2_cores"]
        usd = INSTANCE_USD_PER_HOUR / (rps * 3600) * 1000
        served = {
            "label": f"ModernBERT-base, int8 ONNX served (seed {onnx['model'].rsplit('seed', 1)[1]})",
            "runs": 1,
            "acc": [onnx["int8"]["accuracy"]],
            "f1": [onnx["int8"]["macro_f1"]],
            "p50": f"{latency['int8']['in_process']['threads_2']['p50']:.1f} ms (ONNX)",
            "cost": dollars(usd),
            "trained": "—",
        }
        rows.insert(main_families.index("modernbert") + 1 if "modernbert" in main_families else 1, served)

    out = ["# Results", "", "Generated by `scripts/report.py` from `results/`. Test set: the official Banking77"]
    out += ["test split, 3,080 messages, 40 per intent. Rows with several runs are mean ± standard deviation over"]
    out += ["seeds (same data split, different initialization and data order).", ""]
    out += [
        "| Approach | Runs | Test accuracy | Macro-F1 | p50 latency, 1 message | $ per 1K requests | Trained params |"
    ]
    out += ["|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['label']} | {r['runs']} | {spread(r['acc'])} | {spread_f1(r['f1'])} | {r['p50']} | {r['cost']} | "
            f"{r['trained']} |"
        )
    out += [
        "",
        "Latency is on the CPU with 2 threads (Apple M5 Pro) for the local models and the API round trip for",
        f"Claude (8 requests in flight). Self-hosted cost assumes {INSTANCE} at ${INSTANCE_USD_PER_HOUR}/hour",
        "running at the 2-core throughput measured over HTTP below; a cloud vCPU is slower than an M5 core, so",
        "read it as an order of magnitude. Claude's cost is the token usage the API reported, at list prices.",
    ]
    cpus = [read(p) for p in sorted(SERVING.glob("cpu-*.json"))]
    x86 = [c for c in cpus if c["machine"] == "x86_64"]
    if served and x86:
        out += [
            f"The int8 row's accuracy is on the Apple CPU. The same file scores {pct(x86[0]['int8_7bit']['accuracy'])} "
            "on an x86 server (see the export section): int8 accuracy depends on the CPU's kernels; float32's doesn't.",
        ]

    if encoder_seeds or lora:
        out += ["", "## Every training run", ""]
        out += ["| Run | Val macro-F1 | Test accuracy | Test macro-F1 | Best epoch | Train time (MPS) |"]
        out += ["|---|---|---|---|---|---|"]
        for run in sorted((r for r in runs if r["approach"] in ("encoder", "lora")), key=lambda r: r["name"]):
            out.append(
                f"| {run['name']} | {run['val_macro_f1']:.4f} | {pct(run['accuracy'])} | {run['macro_f1']:.4f} | "
                f"{run['best_epoch']:.0f} | {run['train_seconds'] / 60:.0f} min |"
            )

    if sweep:
        out += ["", "## LoRA rank sweep (seed 0)", ""]
        out += ["The rank is chosen on validation macro-F1; test accuracy is shown for reference only.", ""]
        out += ["| Rank | Trainable params | Val macro-F1 | Test accuracy |", "|---|---|---|---|"]
        for fam, run in sweep:
            mark = " (chosen)" if fam == chosen_lora else ""
            out.append(
                f"| {run['rank']}{mark} | {run['trainable_params'] / 1e6:.1f}M | {run['val_macro_f1']:.4f} | "
                f"{pct(run['accuracy'])} |"
            )

    def seed0(fam: str) -> dict | None:
        rs = by_family.get(fam, [])
        return next((r for r in rs if r.get("seed", 0) == 0), None)

    def right(run: dict) -> list[bool]:
        return [p == t for p, t in zip(run["predictions"], truth, strict=True)]

    pairs = [
        ("modernbert", "tfidf-logreg"),
        (chosen_lora, "modernbert"),
        ("modernbert", "claude-haiku-5-5-10shot-high"),
        ("claude-haiku-5-5-10shot-high", "claude-haiku-5-5-0shot-high"),
    ]
    tests = [(a, b, seed0(a), seed0(b)) for a, b in pairs if a and seed0(a) and seed0(b)]
    if tests:
        out += ["", "## Are the differences real? (paired McNemar tests, seed 0 runs)", ""]
        out += ["| A | B | A right, B wrong | B right, A wrong | p-value |", "|---|---|---|---|---|"]
        for a, b, ra, rb in tests:
            ca, cb = right(ra), right(rb)
            only_a = sum(x and not y for x, y in zip(ca, cb, strict=True))
            only_b = sum(y and not x for x, y in zip(ca, cb, strict=True))
            out.append(f"| {label(a)} | {label(b)} | {only_a} | {only_b} | {mcnemar(ca, cb):.2g} |")

    claude = seed0("claude-haiku-5-5-10shot-high")
    if claude:
        # One seed can land on either side of p = 0.05, so every fine-tuned run is tested against Claude.
        contenders = [
            (r["name"], r) for r in sorted(encoder_seeds + by_family.get(chosen_lora, []), key=lambda r: r["name"])
        ]
        if onnx:
            contenders.append((f"{onnx['model']}, int8 ONNX (served)", onnx["int8"]))
        out += ["", f"Every fine-tuned run against {label('claude-haiku-5-5-10shot-high')}:", ""]
        out += ["| Run | Test accuracy | Run right, Claude wrong | Claude right, run wrong | p-value |"]
        out += ["|---|---|---|---|---|"]
        cb = right(claude)
        for name, run in contenders:
            ca = right(run)
            only_a = sum(x and not y for x, y in zip(ca, cb, strict=True))
            only_b = sum(y and not x for x, y in zip(ca, cb, strict=True))
            out.append(f"| {name} | {pct(run['accuracy'])} | {only_a} | {only_b} | {mcnemar(ca, cb):.2g} |")

    if onnx:
        agree = sum(x == y for x, y in zip(onnx["fp32"]["predictions"], onnx["int8"]["predictions"], strict=True))
        out += ["", f"## Export: {onnx['model']} to ONNX, then int8", ""]
        out += ["| File | Size | Test accuracy | Macro-F1 |", "|---|---|---|---|"]
        for key in ("fp32", "int8"):
            e = onnx[key]
            out.append(f"| {e['file']} | {e['size_mb']:.0f} MB | {pct(e['accuracy'])} | {e['macro_f1']:.4f} |")
        out += ["", f"int8 and float32 give the same answer on {agree:,} of 3,080 test messages."]
        variants = read(SERVING / f"quant-variants-{onnx['model']}.json")
        if variants:
            out += ["", "int8 variants, chosen on validation (`scripts/quant_variants.py`):", ""]
            out += [
                "| Variant | Size | Val accuracy | Val macro-F1 | p50, 1 thread | Can overflow on x86 without VNNI |",
                "|---|---|---|---|---|---|",
            ]
            fp32 = variants["fp32"]
            out.append(
                f"| float32 | — | {pct(fp32['accuracy'])} | {fp32['macro_f1']:.4f} | "
                f"{fp32['p50_ms_1_thread']:.1f} ms | no |"
            )
            for key, title in (
                ("per_tensor", "int8, per-tensor"),
                ("per_channel", "int8, per-channel"),
                ("per_channel_reduce_range", "int8, per-channel, 7-bit weights (shipped)"),
                ("per_tensor_u8u8", "int8, per-tensor, unsigned weights"),
                ("per_channel_u8u8", "int8, per-channel, unsigned weights"),
            ):
                v = variants[key]
                overflow = "yes" if v["clamps_on_x86_without_vnni"] else "no"
                out.append(
                    f"| {title} | {v['size_mb']:.0f} MB | {pct(v['accuracy'])} | {v['macro_f1']:.4f} | "
                    f"{v['p50_ms_1_thread']:.1f} ms | {overflow} |"
                )

        live, first_live = read(SERVING / "live-demo.json"), read(SERVING / "live-demo-first-int8.json")
        if cpus:
            out += ["", "### The same files on other CPUs (test accuracy)", ""]
            out += ["`scripts/cpu_check.py` scores the published files on the machine it runs on (the x86 run was in a"]
            out += [
                "Vercel Sandbox); `scripts/check_live.py` scores the live demo (a Vercel Function, x86).",
                "",
            ]
            columns = [f"{c['cpu']} ({', '.join(c['features'])})" for c in cpus]
            out += ["| File | " + " | ".join(columns) + " | Live demo |", "|---|" + "---|" * (len(cpus) + 1)]
            rows_ = (
                ("float32", "float32", None),
                ("int8_8bit", "int8, 8-bit weights (first export)", first_live),
                ("int8_7bit", "int8, 7-bit weights (shipped)", live),
                ("int8_7bit_no_fusions", "int8, 7-bit weights, graph fusions off", None),
                ("int8_unsigned", "int8, unsigned weights", None),
            )
            for key, title, demo in rows_:
                cells = [pct(c[key]["accuracy"]) for c in cpus]
                out.append(f"| {title} | " + " | ".join(cells) + f" | {pct(demo['accuracy']) if demo else '—'} |")
            if len(cpus) == 2:
                same = sum(a == b for a, b in zip(*(c["float32"]["predictions"] for c in cpus), strict=True))
                out += ["", f"float32 gives the same answer on both CPUs for {same:,} of 3,080 messages."]
                out += ["int8 does not: the integer kernels, and how each CPU fuses them, differ by platform."]

    if latency:
        out += ["", f"## CPU latency and throughput ({latency['machine']})", ""]
        out += ["| Measure | float32 | int8 | int8 speed-up |", "|---|---|---|---|"]

        def line(title: str, get, unit: str, higher_is_better: bool = False) -> None:
            a, b = get(latency["fp32"]), get(latency["int8"])
            ratio = b / a if higher_is_better else a / b
            digits = 0 if unit == "req/s" else 1
            out.append(f"| {title} | {a:,.{digits}f} {unit} | {b:,.{digits}f} {unit} | {ratio:.1f}x |")

        for threads, title in (("1", "1 thread"), ("2", "2 threads"), ("all", "all cores")):
            line(
                f"p50, in process, {title}",
                lambda d, t=threads: d["in_process"][f"threads_{t}"]["p50"],
                "ms",
            )
        line("p95, in process, 2 threads", lambda d: d["in_process"]["threads_2"]["p95"], "ms")
        line("p50 over HTTP (FastAPI, 2 threads)", lambda d: d["http"]["p50"], "ms")
        line("p95 over HTTP (FastAPI, 2 threads)", lambda d: d["http"]["p95"], "ms")
        line(
            "Requests/s over HTTP, 2 cores (1 thread x 2 clients)",
            lambda d: d["http"]["requests_per_second_2_cores"],
            "req/s",
            True,
        )
        out.append(
            f"| Model file | {latency['fp32']['size_mb']:.0f} MB | {latency['int8']['size_mb']:.0f} MB | "
            f"{latency['fp32']['size_mb'] / latency['int8']['size_mb']:.1f}x smaller |"
        )

    best = seed0("modernbert")
    few = seed0("claude-haiku-5-5-10shot-high")
    if best:
        out += ["", "## Error analysis: the intents each model mixes up most (seed 0)", ""]
        for title, run in (("ModernBERT-base", best), ("Claude Haiku 5.5, 10-shot", few)):
            if not run:
                continue
            out += [f"**{title}** ({3080 - sum(right(run))} errors)", ""]
            out += ["| Intent A | Intent B | Errors | A read as B | B read as A |", "|---|---|---|---|---|"]
            for p in most_confused(truth, run["predictions"], names, k=10):
                out.append(f"| {p['a']} | {p['b']} | {p['errors']} | {p['a_as_b']} | {p['b_as_a']} |")
            out.append("")
        if few:
            both = sum(not x and not y for x, y in zip(right(best), right(few), strict=True))
            out.append(f"Both models are wrong on {both} messages; see README.md for what they have in common.")

    (ROOT / "RESULTS.md").write_text("\n".join(out).rstrip() + "\n")
    chart(rows)
    print("wrote RESULTS.md and docs/accuracy.png")


def chart(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, mark = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    order = sorted(rows, key=lambda r: statistics.mean(r["acc"]))
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(order) + 1.4), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for y, r in enumerate(order):
        mean = 100 * statistics.mean(r["acc"])
        sd = 100 * statistics.stdev(r["acc"]) if len(r["acc"]) > 1 else 0
        if sd:
            ax.plot([mean - sd, mean + sd], [y, y], color=mark, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([mean], [y], s=64, color=mark, edgecolor=surface, linewidth=2, zorder=3)
        ax.annotate(
            f"{mean:.1f}%", (mean, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=ink
        )
        side = [f"{r['cost']} per 1K" if r["cost"] != "—" else "", r["p50"] if r["p50"] != "—" else ""]
        ax.annotate(
            " · ".join(part for part in side if part),
            (1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=muted,
        )
    ax.set_yticks(range(len(order)), [r["label"] for r in order], fontsize=9, color=ink)
    lo = min(100 * min(r["acc"]) for r in order)
    ax.set_xlim(5 * int(lo // 5) - 1, 96)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.xaxis.grid(True, color=grid, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Test accuracy on Banking77 (%), line = ± 1 sd over seeds", fontsize=8, color=muted)
    fig.suptitle("Fine-tuned small models vs Claude on 77 banking intents", x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "accuracy.png", facecolor=surface)


if __name__ == "__main__":
    main()
