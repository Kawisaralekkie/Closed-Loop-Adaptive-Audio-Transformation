#!/usr/bin/env python3
"""Plot comparison across LLM pipeline variants.

Usage:
    python3 scripts/plot_comparison_llm.py
"""

from __future__ import annotations
import csv, json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
})

# ── Config ──
# SS-ENABLED rerun (2026-07-15): source separation is wired to the LLM agents
# and the prompt de-bias is active, so llm_* configs can actually select SS.
# NOTE: llm_no_memory_no_ss was not downloaded into this folder — add it back
# (download from S3 reports/llm_no_memory_no_ss/...) to complete the 2x2 grid.
# Default is the OneDrive path (runs exactly as before). If OneDrive keeps
# timing out, copy the folder to a local disk and set RUN_DIR to point there:
#   cp -R "logs/s3/20260715_final" ~/csm_reports/final
#   RUN_DIR=~/csm_reports/final ~/csm_analysis_venv/bin/python scripts/plot_comparison_llm_v2.py
_RUN = os.environ.get("RUN_DIR", "logs/s3/20260715_final")
VERSIONS = {
    "LLM-Mem-SS": {          # LLM, memory ON,  source separation ON
        "path": f"{_RUN}/llm_with_memory",
        "color": "#2E7D32",  # green
    },
    "LLM-Mem-NoSS": {        # LLM, memory ON,  source separation OFF
        "path": f"{_RUN}/llm_with_memory_no_ss",
        "color": "#F9A825",  # amber/gold
    },
    "LLM-NoMem-SS": {        # LLM, memory OFF, source separation ON
        "path": f"{_RUN}/llm_no_memory",
        "color": "#6A1B9A",  # purple
    },
    "Rule-SS": {             # Rule-based, source separation ON (pre-processing)
        "path": f"{_RUN}/rule_based_ss",
        "color": "#D32F2F",       # RED line/points
        "star_color": "#F48FB1",  # PINK for SS stars
    },
}

OUT_DIR = "plots/comparison_llm_paper"


def load(folder):
    if not folder or not os.path.exists(folder):
        return []
    r = []
    for f in sorted(glob.glob(os.path.join(folder, "*_report.json"))):
        with open(f) as fh:
            r.append(json.load(fh))
    return r


def sc(reports):
    return [c for r in reports for c in r.get("chunks", [])
            if c.get("had_speech") and c.get("metrics")]


def no_sc(reports):
    return [c for r in reports for c in r.get("chunks", [])
            if not c.get("had_speech") and c.get("metrics")]


# ─────────────────────────────────────────────────────────────────────────────
# [v2 PATCH] Recompute PreserveScore as (TC@3 + TA@1) / 2
#   • TC@3 = |top3_orig ∩ top3_proc| / 3      (Top-3 Consistency)
#   • TA@1 = 1 if top-1 labels match else 0   (Top-1 Agreement)
# This replaces the old confidence-based preserve_score embedded in JSON.
# Use semantic_preserve_score(c) everywhere instead of c["metrics"]["utility"]["preserve_score"].
# ─────────────────────────────────────────────────────────────────────────────

def semantic_preserve_score(c):
    """(TC@3 + TA@1) / 2 from a chunk's classification top3 vs original top3.
    Returns None if classification info is missing."""
    orig = c.get("classification_top3_original", [])
    proc = c.get("classification_top3", [])
    if not orig or not proc:
        return None
    set_a = set(p["label"] for p in orig[:3])
    set_b = set(p["label"] for p in proc[:3])
    tc3 = len(set_a & set_b) / 3.0
    ta1 = 1.0 if orig[0]["label"] == proc[0]["label"] else 0.0
    return (tc3 + ta1) / 2.0


def preserve_list(chunks):
    """Filter and return non-None preserve scores."""
    out = [semantic_preserve_score(c) for c in chunks]
    return [v for v in out if v is not None]


def annotate_bar(ax, bar, text, ylim_top):
    """Put text above bar, clamped below title."""
    y = min(bar.get_height() + 0.03, ylim_top - 0.12)
    ax.text(bar.get_x() + bar.get_width()/2, y, text,
            ha="center", va="bottom", fontweight="bold", fontsize=14)


def chunk_used_ss(c):
    """True if a chunk used source separation, via EITHER mechanism:
      • Adaptive/rule-based: SS pre-processing → flag `used_source_separation`.
      • LLM agents: SS selected as a recipe → `recipe_applied.recipe_name`.
    The two agents record SS differently, so both must be checked.
    """
    if c.get("used_source_separation"):
        return True
    return (c.get("recipe_applied") or {}).get("recipe_name") == "RECIPE_SOURCE_SEPARATION"


# ═══════════════════════════════════════════════════════════════════════════
# [v2 ADD] Thesis evidence: (1) statistical Memory-vs-NoMemory test + CI,
#          (2) LLM run profile (Claude Haiku 4.5 prompt/temp/cost/latency).
# ═══════════════════════════════════════════════════════════════════════════

# LLM decoding config (from src/agents/llm_privacy_control_agent.py::_call_bedrock)
LLM_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512
# Amazon Bedrock price for Claude Haiku 4.5 (USD / 1M tokens): $1.10 in, $5.50 out.
# Direct Anthropic API is $1.00 / $5.00 — adjust if not routed through Bedrock.
HAIKU45_PRICE_IN_PER_MTOK = 1.10
HAIKU45_PRICE_OUT_PER_MTOK = 5.50


def _detect_memory_labels(labels):
    """Split configuration labels into (memory, no_memory) groups by name."""
    mem, nomem = [], []
    for l in labels:
        ll = l.lower()
        is_nomem = ("nomem" in ll) or ("no_memory" in ll) or ("no-mem" in ll)
        if is_nomem:
            nomem.append(l)
        elif ("mem" in ll) or ("memory" in ll):
            mem.append(l)
    return mem, nomem


def _bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI for the mean. Returns (mean, lo, hi)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def _cliffs_delta(a, b):
    """Cliff's delta effect size. |d|: .11 negligible, .28 small, .43 medium."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (a.size * b.size))


def _stars(p):
    if p is None or p != p:
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def stats_memory_vs_nomemory(paper_chunks, out_dir):
    """Task 1 — Memory vs No-Memory: 95% CI + significance test per metric.

    Writes ``statistical_memory_vs_nomemory.csv`` and a bar figure with 95% CI
    error bars and significance stars. Uses a PAIRED test (Wilcoxon signed-rank)
    on chunks present in both configs; falls back to Mann-Whitney U (unpaired).
    """
    labels = list(paper_chunks.keys())
    mem_labels, nomem_labels = _detect_memory_labels(labels)
    if not mem_labels or not nomem_labels:
        print("  ⚠ stats: need both a Memory and a No-Memory config — skipped")
        return

    def _pick(cands):  # prefer a matched pair with SS enabled on both sides
        ss = [c for c in cands if "-ss" in c.lower() or c.lower().endswith("ss")]
        ss = [c for c in ss if "no_ss" not in c.lower() and "noss" not in c.lower()]
        return (ss or cands)[0]
    mem_label, nomem_label = _pick(mem_labels), _pick(nomem_labels)

    try:
        from scipy import stats as _sps
        have_scipy = True
    except Exception:
        have_scipy = False

    metrics = [
        ("privacy_score", "Privacy Score"),
        ("preserve_new", "Utility (TC@3+TA@1)/2"),
        ("trials_used", "Trials per chunk"),
        ("used_ss", "Source-separation rate"),
    ]

    def _key(r):
        return (r.get("source_id", ""), r.get("chunk_id", ""))

    def _val(r, k):
        v = r.get(k)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        return None if v is None else float(v)

    mem_by = {_key(r): r for r in paper_chunks[mem_label]}
    nomem_by = {_key(r): r for r in paper_chunks[nomem_label]}
    shared = [k for k in mem_by if k in nomem_by]
    paired = len(shared) > 0

    rows = []
    for key, disp in metrics:
        if paired:
            pairs = [(_val(mem_by[k], key), _val(nomem_by[k], key)) for k in shared]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            a = [p[0] for p in pairs]; b = [p[1] for p in pairs]
            diffs = [x - y for x, y in pairs]
            test = "Wilcoxon signed-rank (paired)"
            if have_scipy and diffs and any(d != 0 for d in diffs):
                try:
                    stat, pval = map(float, _sps.wilcoxon(a, b))
                except Exception:
                    stat, pval = float("nan"), float("nan")
            else:
                stat, pval = float("nan"), float("nan")
            dmean, dlo, dhi = _bootstrap_ci(diffs)
        else:
            a = [v for v in (_val(r, key) for r in paper_chunks[mem_label]) if v is not None]
            b = [v for v in (_val(r, key) for r in paper_chunks[nomem_label]) if v is not None]
            test = "Mann-Whitney U (unpaired)"
            if have_scipy and a and b:
                stat, pval = map(float, _sps.mannwhitneyu(a, b, alternative="two-sided"))
            else:
                stat, pval = float("nan"), float("nan")
            rng = np.random.default_rng(0)
            av, bv = np.asarray(a), np.asarray(b)
            if av.size and bv.size:
                boots = (rng.choice(av, (10000, av.size)).mean(1)
                         - rng.choice(bv, (10000, bv.size)).mean(1))
                dmean = float(av.mean() - bv.mean())
                dlo, dhi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            else:
                dmean = dlo = dhi = float("nan")
        m_mean, m_lo, m_hi = _bootstrap_ci(a)
        n_mean, n_lo, n_hi = _bootstrap_ci(b)
        rows.append({
            "metric": disp, "n": len(a), "test": test,
            "mem_mean": m_mean, "mem_lo": m_lo, "mem_hi": m_hi,
            "nomem_mean": n_mean, "nomem_lo": n_lo, "nomem_hi": n_hi,
            "diff": dmean, "diff_lo": dlo, "diff_hi": dhi,
            "stat": stat, "p": pval, "cliffs_delta": _cliffs_delta(a, b),
        })

    # ── CSV ──
    csv_path = os.path.join(out_dir, "statistical_memory_vs_nomemory.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["metric", "memory_config", "nomemory_config", "n", "test",
                       "mem_mean", "mem_ci95_low", "mem_ci95_high",
                       "nomem_mean", "nomem_ci95_low", "nomem_ci95_high",
                       "mean_diff(mem-nomem)", "diff_ci95_low", "diff_ci95_high",
                       "statistic", "p_value", "cliffs_delta", "significant_p<0.05"])
        for r in rows:
            wcsv.writerow([r["metric"], mem_label, nomem_label, r["n"], r["test"],
                           f"{r['mem_mean']:.4f}", f"{r['mem_lo']:.4f}", f"{r['mem_hi']:.4f}",
                           f"{r['nomem_mean']:.4f}", f"{r['nomem_lo']:.4f}", f"{r['nomem_hi']:.4f}",
                           f"{r['diff']:.4f}", f"{r['diff_lo']:.4f}", f"{r['diff_hi']:.4f}",
                           f"{r['stat']:.4g}", f"{r['p']:.4g}", f"{r['cliffs_delta']:.4f}",
                           "yes" if (r["p"] == r["p"] and r["p"] < 0.05) else "no"])
    print(f"  ✓ statistical_memory_vs_nomemory.csv  ({mem_label} vs {nomem_label}, "
          f"{'paired n=%d' % len(shared) if paired else 'unpaired'})")

    # ── Figure: mem vs nomem grouped bars with 95% CI + significance stars ──
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 7))
    if len(metrics) == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        gx = np.arange(2)
        means = [r["mem_mean"], r["nomem_mean"]]
        errs = [[r["mem_mean"] - r["mem_lo"], r["nomem_mean"] - r["nomem_lo"]],
                [r["mem_hi"] - r["mem_mean"], r["nomem_hi"] - r["nomem_mean"]]]
        bars = ax.bar(gx, means, 0.6, yerr=errs, capsize=8,
                      color=["#2E7D32", "#C62828"], alpha=0.85)
        ax.set_xticks(gx); ax.set_xticklabels(["Memory", "No-Memory"], fontsize=16)
        ax.set_title(f"{r['metric']}\n{_stars(r['p'])} (p={r['p']:.3g})", fontsize=15)
        ax.grid(axis="y", alpha=0.3)
        top = max(r["mem_hi"], r["nomem_hi"])
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{m:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=13)
        # significance bracket
        y = top * 1.08 if top > 0 else 0.1
        ax.plot([0, 0, 1, 1], [y, y * 1.02, y * 1.02, y], lw=1.4, color="black")
        ax.text(0.5, y * 1.03, _stars(r["p"]), ha="center", va="bottom",
                fontweight="bold", fontsize=16)
    fig.suptitle(f"Memory vs No-Memory — 95% CI + significance ({rows[0]['test']})",
                 fontweight="bold", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "statistical_memory_vs_nomemory.png"), dpi=150)
    plt.close(fig)
    print("  ✓ statistical_memory_vs_nomemory.png")


def llm_run_profile(data, labels, out_dir):
    """Task 2 — per-config LLM profile: tokens, cost, latency (Claude Haiku 4.5)."""
    rows = []
    for l in labels:
        in_tok = out_tok = n_calls = n_speech = 0
        latencies, trials = [], []
        runtime = 0.0
        for d in data[l]["reports"]:
            tt = d.get("total_llm_token_usage") or {}
            runtime += d.get("total_runtime_seconds") or 0.0
            c_in = c_out = 0
            for c in d.get("chunks", []):
                u = c.get("llm_token_usage") or {}
                c_in += u.get("input_tokens") or 0
                c_out += u.get("output_tokens") or 0
                if c.get("had_speech"):
                    n_speech += 1
                    if c.get("trials") is not None:
                        trials.append(c["trials"])
                for resp in (c.get("llm_responses") or []):
                    n_calls += 1
                    if resp.get("latency_ms") is not None:
                        latencies.append(resp["latency_ms"])
            in_tok += tt.get("input_tokens") if tt.get("input_tokens") is not None else c_in
            out_tok += tt.get("output_tokens") if tt.get("output_tokens") is not None else c_out
        if n_calls == 0:
            continue  # non-LLM config (e.g. rule-based)
        cost = (in_tok / 1e6) * HAIKU45_PRICE_IN_PER_MTOK + (out_tok / 1e6) * HAIKU45_PRICE_OUT_PER_MTOK
        rows.append({
            "config": l, "n_llm_calls": n_calls, "speech_chunks": n_speech,
            "mean_trials": float(np.mean(trials)) if trials else 0.0,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok, "cost_usd": cost,
            "cost_per_speech_chunk": cost / n_speech if n_speech else 0.0,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        })
    if not rows:
        print("  ⚠ llm_run_profile: no LLM token data found — skipped")
        return
    path = os.path.join(out_dir, "llm_run_profile.csv")
    with open(path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["config", "model", "temperature", "max_tokens",
                       "n_llm_calls", "speech_chunks", "mean_trials",
                       "input_tokens", "output_tokens", "total_tokens",
                       "cost_usd", "cost_per_speech_chunk_usd",
                       "mean_latency_ms", "p50_latency_ms", "p95_latency_ms"])
        for r in rows:
            wcsv.writerow([r["config"], LLM_MODEL_ID, LLM_TEMPERATURE, LLM_MAX_TOKENS,
                           r["n_llm_calls"], r["speech_chunks"], f"{r['mean_trials']:.3f}",
                           r["input_tokens"], r["output_tokens"], r["total_tokens"],
                           f"{r['cost_usd']:.4f}", f"{r['cost_per_speech_chunk']:.6f}",
                           f"{r['mean_latency_ms']:.1f}", f"{r['p50_latency_ms']:.1f}",
                           f"{r['p95_latency_ms']:.1f}"])
    print(f"  ✓ llm_run_profile.csv  ({len(rows)} LLM configs; "
          f"model={LLM_MODEL_ID}, temp={LLM_TEMPERATURE})")


def plot_tradeoff_box(data, labels, out_dir):
    """Cleaner alternative to the scatter trade-off: two side-by-side box plots
    (Privacy | Utility) per config. Box = median + IQR + whiskers + outliers,
    so per-chunk spread is visible without thousands of overlapping points."""
    priv_by = {l: [c["metrics"]["privacy"]["privacy_score"] for c in sc(data[l]["reports"])]
               for l in labels}
    util_by = {l: preserve_list(sc(data[l]["reports"])) for l in labels}

    fig, (axp, axu) = plt.subplots(1, 2, figsize=(20, 9))
    pos = np.arange(len(labels)) + 1

    def _draw(ax, series, title, thr, thr_txt):
        vals = [series[l] for l in labels]
        bp = ax.boxplot(vals, positions=pos, widths=0.6, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                        markeredgecolor="black", markersize=9),
                        medianprops=dict(color="black", lw=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.25))
        for patch, l in zip(bp["boxes"], labels):
            patch.set_facecolor(data[l]["color"]); patch.set_alpha(0.65)
        # mean value annotation above each box
        for i, l in enumerate(labels):
            if series[l]:
                m = float(np.mean(series[l]))
                ax.text(pos[i], 1.03, f"{m:.3f}", ha="center", va="bottom",
                        fontsize=16, fontweight="bold", color=data[l]["color"])
        ax.axhline(thr, color="#B71C1C", ls="--", lw=2, alpha=0.7)
        ax.text(len(labels) + 0.4, thr, thr_txt, color="#B71C1C", fontsize=15,
                fontweight="bold", va="center", ha="right")
        ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.12); ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    _draw(axp, priv_by, "Privacy Score per chunk", 0.65, "T_p = 0.65")
    axp.set_ylabel("Privacy Score")
    _draw(axu, util_by, "Utility Score per chunk\n(TC@3+TA@1)/2", 0.80, "T_u = 0.80")
    axu.set_ylabel("Utility Score")

    fig.suptitle("Privacy vs Utility — per-chunk distribution "
                 "(box = median/IQR/whiskers; ◇ = mean)", fontweight="bold", fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "privacy_utility_box.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ privacy_utility_box.png")


def plot_speechratio_bars(data, labels, out_dir):
    """Cleaner alternative to the SR-vs-privacy scatter: grouped bar chart with
    speech-ratio bins on the x-axis. Bars = mean privacy per config per bin,
    with 95% CI error bars, per-bar n, and an SS marker where SS was applied."""
    bins = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0001)]
    bin_lbls = ["SR 0–0.3", "SR 0.3–0.6", "SR 0.6–1.0"]
    nb, nl = len(bins), len(labels)
    bw = 0.8 / nl
    xb = np.arange(nb)

    fig, ax = plt.subplots(figsize=(17, 10))
    for j, l in enumerate(labels):
        s = sc(data[l]["reports"])
        means, errs, ns, ss_flags = [], [], [], []
        for lo, hi in bins:
            grp = [c for c in s if lo <= c.get("speech_ratio", 0) < hi]
            pv = [c["metrics"]["privacy"]["privacy_score"] for c in grp]
            n_ss = sum(1 for c in grp if chunk_used_ss(c))
            if pv:
                m, clo, chi = _bootstrap_ci(pv)
                means.append(m); errs.append(max(m - clo, chi - m))
            else:
                means.append(0.0); errs.append(0.0)
            ns.append(len(pv)); ss_flags.append(n_ss)
        offs = xb + j * bw - (nl - 1) * bw / 2
        bars = ax.bar(offs, means, bw, yerr=errs, capsize=5,
                      color=data[l]["color"], alpha=0.85, label=l,
                      error_kw=dict(lw=1.5))
        for b, m, n, nss in zip(bars, means, ns, ss_flags):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.015, f"n={n}",
                    ha="center", va="bottom", fontsize=12)
            if nss > 0:  # star marks bars where source separation was applied
                ax.text(b.get_x() + b.get_width() / 2, 0.02, f"★{nss}",
                        ha="center", va="bottom", fontsize=13, color="black",
                        fontweight="bold")

    ax.axhline(0.65, color="#E65100", ls="--", lw=2.5, alpha=0.9)
    ax.text(nb - 0.5, 0.66, "privacy threshold  T_p = 0.65", color="#E65100",
            fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax.set_xticks(xb); ax.set_xticklabels(bin_lbls)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Speech Ratio bin"); ax.set_ylabel("Mean Privacy Score (95% CI)")
    ax.set_title("Privacy Score by Speech-Ratio bin\n"
                 "(bars = mean ± 95% CI; ★n = chunks using source separation)",
                 fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=nl,
              frameon=True, edgecolor="black")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(out_dir, "speech_ratio_privacy_bars.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ speech_ratio_privacy_bars.png")


def _priv_util_pairs(reports):
    """Return (privacy, utility) pairs for speech chunks that have top-3 info."""
    out = []
    for c in sc(reports):
        u = semantic_preserve_score(c)
        if u is None:
            continue
        out.append((c["metrics"]["privacy"]["privacy_score"], u))
    return out


def _grid(n):
    """Rows, cols for a near-square facet grid (n<=4 -> 2x2)."""
    if n <= 2:
        return 1, n
    if n <= 4:
        return 2, 2
    cols = int(np.ceil(np.sqrt(n)))
    return int(np.ceil(n / cols)), cols


def _pareto_frontier(centroids):
    """Non-dominated frontier over (privacy, utility) centroids — maximise both.
    Returns sorted (x, y) points, or [] if fewer than 2 points."""
    pts = sorted(centroids.values(), key=lambda p: p[0], reverse=True)
    frontier, best_u = [], -1.0
    for px, py in pts:
        if py >= best_u:
            frontier.append((px, py)); best_u = py
    return sorted(frontier) if len(frontier) >= 2 else []


def plot_tradeoff_density(data, labels, out_dir):
    """(1) 2D density (hexbin) facets for Privacy vs Utility — shows WHERE the
    chunks concentrate instead of a solid blob of overplotted points. The
    cross-config Pareto frontier is overlaid on every facet for reference."""
    # Centroids + Pareto frontier across ALL configs (shown on each facet).
    centroids = {}
    for l in labels:
        pairs = _priv_util_pairs(data[l]["reports"])
        if pairs:
            priv, util = zip(*pairs)
            centroids[l] = (float(np.mean(priv)), float(np.mean(util)))
    frontier = _pareto_frontier(centroids)

    nr, nc = _grid(len(labels))
    fig, axes = plt.subplots(nr, nc, figsize=(8 * nc, 7 * nr), squeeze=False)
    for idx, l in enumerate(labels):
        ax = axes[idx // nc][idx % nc]
        pairs = _priv_util_pairs(data[l]["reports"])
        if pairs:
            priv, util = zip(*pairs)
            hb = ax.hexbin(priv, util, gridsize=30, cmap="viridis", mincnt=1,
                           extent=(0, 1, 0, 1))
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("chunk count", fontsize=22)
            cb.ax.tick_params(labelsize=22)
        # Pareto frontier + all centroids (reference), current mode's star large.
        if frontier:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, color="white", ls="-", lw=4.0, alpha=0.9, zorder=4)
            ax.plot(fx, fy, color="black", ls="-", lw=2.2, alpha=0.95, zorder=5,
                    label="Pareto frontier")
        for ll, (px, py) in centroids.items():
            if ll == l:
                continue
            ax.scatter([px], [py], s=90, color=data[ll]["color"], marker="*",
                       edgecolors="white", linewidths=0.8, alpha=0.85, zorder=6)
        if l in centroids:
            cx, cy = centroids[l]
            ax.scatter([cx], [cy], s=420, color=data[l]["color"], marker="*",
                       edgecolors="white", linewidths=1.8, zorder=7)
            ax.text(cx, cy + 0.05, f"mean\n({cx:.2f}, {cy:.2f})", ha="center",
                    va="bottom", fontsize=22, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=data[l]["color"], alpha=0.85),
                    zorder=8)
        ax.axvline(0.65, color="#E65100", ls="--", lw=2, alpha=0.8)
        ax.axhline(0.80, color="#B71C1C", ls="--", lw=2, alpha=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=22)
        ax.set_title(l, fontweight="bold", fontsize=24, color=data[l]["color"])
        ax.set_xlabel("Privacy Score", fontsize=22)
        ax.set_ylabel("Utility (TC@3+TA@1)/2", fontsize=22)
        if frontier and idx == 0:
            ax.legend(loc="lower left", fontsize=24, frameon=True, edgecolor="black")
    # hide any spare axes
    for k in range(len(labels), nr * nc):
        axes[k // nc][k % nc].axis("off")
    fig.suptitle("Privacy vs Utility — 2D density (hexbin) per config\n"
                 "(brighter = more chunks; star = mean; solid = Pareto frontier; "
                 "dashed = thresholds)", fontweight="bold", fontsize=24)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "privacy_utility_density.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ privacy_utility_density.png")


def plot_tradeoff_alpha_scatter(data, labels, out_dir):
    """(3) Simple low-alpha scatter — dense regions darken naturally. Kept as a
    quick, familiar alternative to the density map."""
    fig, ax = plt.subplots(figsize=(14, 10))
    for l in labels:
        pairs = _priv_util_pairs(data[l]["reports"])
        if pairs:
            priv, util = zip(*pairs)
            ax.scatter(priv, util, s=18, alpha=0.12, color=data[l]["color"],
                       edgecolors="none", label=l)
    for l in labels:
        pairs = _priv_util_pairs(data[l]["reports"])
        if pairs:
            priv, util = zip(*pairs)
            ax.scatter([np.mean(priv)], [np.mean(util)], s=360, marker="*",
                       color=data[l]["color"], edgecolors="black", linewidths=1.4,
                       zorder=6, label=f"{l} (mean)")
    ax.axvline(0.65, color="#E65100", ls="--", lw=2, alpha=0.7, label="T_p = 0.65")
    ax.axhline(0.80, color="#B71C1C", ls="--", lw=2, alpha=0.7, label="T_u = 0.80")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Privacy Score"); ax.set_ylabel("Utility Score (TC@3+TA@1)/2")
    ax.set_title("Privacy vs Utility Trade-off — low-alpha scatter\n"
                 "(darker = denser; ★ = per-config mean)", fontweight="bold")
    handles, lbls = ax.get_legend_handles_labels()
    seen = {}
    for h, lb in zip(handles, lbls):
        seen.setdefault(lb, h)
    ax.legend(seen.values(), seen.keys(), loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=True, edgecolor="black", fontsize=18)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "privacy_utility_alpha_scatter.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ privacy_utility_alpha_scatter.png")


def plot_speechratio_facets(data, labels, out_dir):
    """(2) Facet / small multiples for Speech Ratio vs Privacy — one subplot per
    config on IDENTICAL x/y scales, so trends are easy to compare side by side."""

    def _binned_trend(ratios, priv, n_bins=8, min_n=3):
        if not ratios:
            return [], []
        r = np.asarray(ratios); p = np.asarray(priv)
        edges = np.linspace(0, 1, n_bins + 1)
        cx, cy = [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (r >= lo) & (r < hi if i < n_bins - 1 else r <= hi)
            if mask.sum() >= min_n:
                cx.append((lo + hi) / 2.0); cy.append(float(p[mask].mean()))
        return cx, cy

    nr, nc = _grid(len(labels))
    fig, axes = plt.subplots(nr, nc, figsize=(8 * nc, 6.5 * nr),
                             squeeze=False, sharex=True, sharey=True)
    for idx, l in enumerate(labels):
        ax = axes[idx // nc][idx % nc]
        s = sc(data[l]["reports"])
        r = [c.get("speech_ratio", 0) for c in s]
        p = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        ax.scatter(r, p, s=14, alpha=0.15, color=data[l]["color"], edgecolors="none",
                   zorder=1)
        cx, cy = _binned_trend(r, p, n_bins=8)
        if cx:
            ax.plot(cx, cy, color=data[l]["color"], lw=3.0, marker="o", markersize=8,
                    markeredgecolor="black", markeredgewidth=0.7, zorder=3)
        # SS-applied chunks as stars (pink for Rule via star_color)
        ss_only = [c for c in s if chunk_used_ss(c)]
        if ss_only:
            rs = [c.get("speech_ratio", 0) for c in ss_only]
            ps = [c["metrics"]["privacy"]["privacy_score"] for c in ss_only]
            star_c = data[l].get("star_color", data[l]["color"])
            ax.scatter(rs, ps, s=95, marker="*", color=star_c, edgecolors="black",
                       linewidths=0.7, zorder=4)
        ax.axhline(0.65, color="#E65100", ls="--", lw=2, alpha=0.8, zorder=2)
        ax.axvline(0.30, color="#B71C1C", ls=":", lw=2, alpha=0.8, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.tick_params(axis="both", labelsize=22)
        ax.set_title(l, fontweight="bold", fontsize=24, color=data[l]["color"])
        ax.grid(alpha=0.3)
        if idx % nc == 0:
            ax.set_ylabel("Privacy Score", fontsize=22)
        if idx // nc == nr - 1:
            ax.set_xlabel("Speech Ratio", fontsize=22)
    for k in range(len(labels), nr * nc):
        axes[k // nc][k % nc].axis("off")
    fig.suptitle("Speech Ratio vs Privacy — per-config facets (shared scales)\n"
                 "(line = binned mean; star = source separation applied; "
                 "dashed = T_p 0.65 / SR 0.3)", fontweight="bold", fontsize=24)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, "speech_ratio_facets.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ speech_ratio_facets.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    data = {}
    for label, cfg in VERSIONS.items():
        reports = load(cfg["path"])
        if reports:
            data[label] = {"reports": reports, "color": cfg["color"],
                           "star_color": cfg.get("star_color", cfg["color"])}
            print(f"  {label}: {len(reports)} reports, {len(sc(reports))} speech chunks")
        else:
            print(f"  WARNING: {label} — no data at {cfg['path']}")

    if not data:
        print("ERROR: No data"); sys.exit(1)

    labels = list(data.keys())
    colors = [data[l]["color"] for l in labels]
    x = np.arange(len(labels))
    w = 0.6

    print(f"\nPlotting to {OUT_DIR}/\n")

    # ── 1. Average Privacy Score ──
    means = [np.mean([c["metrics"]["privacy"]["privacy_score"] for c in sc(data[l]["reports"])]) for l in labels]
    stds = [np.std([c["metrics"]["privacy"]["privacy_score"] for c in sc(data[l]["reports"])]) for l in labels]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, means, w, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.3)
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="High (0.65)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Privacy Score"); ax.set_ylim(0, 1.3)
    ax.set_title("Average Privacy Score (±σ)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "avg_privacy_score.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_privacy_score.png")

    # ── 2. Average Utility Score [v2: semantic preserve score] ──
    means = [np.mean(preserve_list(sc(data[l]["reports"]))) for l in labels]
    stds = [np.std(preserve_list(sc(data[l]["reports"]))) for l in labels]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, means, w, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means, stds):
        annotate_bar(ax, bar, f"{m:.4f}±{s:.4f}", 1.1)
    ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7, label="Threshold (0.80)")

    # ── No-speech BASELINE (best-case ceiling) ──
    # Environmental chunks with NO speech are bypassed (not blurred), so their
    # utility is the natural upper bound. We pool no-speech chunks across all
    # configs and draw their mean utility as a reference line.
    ns_scores_all = []
    for l in labels:
        ns_scores_all.extend(preserve_list(no_sc(data[l]["reports"])))
    if ns_scores_all:
        ns_baseline = float(np.mean(ns_scores_all))
        ax.axhline(ns_baseline, color="green", ls="-.", lw=2, alpha=0.85,
                   label=f"No-speech baseline ({ns_baseline:.3f}, n={len(ns_scores_all)})")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Utility Score (TC@3+TA@1)/2"); ax.set_ylim(0, 1.1)
    ax.set_title("Average Utility Score (±σ)\n[Semantic Preservation]", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "avg_utility_score.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_utility_score.png")

    # ── 3. Privacy vs Utility Trade-off [v2: semantic preserve] ──
    fig, ax = plt.subplots(figsize=(14, 10))
    centroids = {}  # label -> (mean_privacy, mean_utility)
    for l in labels:
        s = sc(data[l]["reports"])
        # Pair (privacy, semantic preserve) — drop chunks lacking top3
        pairs = []
        for c in s:
            ps = semantic_preserve_score(c)
            if ps is None:
                continue
            pairs.append((c["metrics"]["privacy"]["privacy_score"], ps))
        if pairs:
            priv, util = zip(*pairs)
            ax.scatter(priv, util, alpha=0.25, s=22, label=l, color=data[l]["color"])
            centroids[l] = (float(np.mean(priv)), float(np.mean(util)))

    # Pareto frontier over the centroids (maximise BOTH privacy and utility).
    pts = sorted(centroids.values(), key=lambda p: p[0], reverse=True)
    frontier, best_u = [], -1.0
    for px, py in pts:                      # sweep high→low privacy
        if py >= best_u:                    # non-dominated (no other point better in both)
            frontier.append((px, py)); best_u = py
    if len(frontier) >= 2:
        fx, fy = zip(*sorted(frontier))
        # Line only (no markers) so the coloured centroid stars stay visible on top.
        ax.plot(fx, fy, color="black", ls="-", lw=3.0, alpha=0.9, zorder=4,
                label="Pareto frontier")

    # Per-config centroids (mean privacy, mean utility) as large stars — label
    # via the legend (not text on the point) so nothing overlaps.
    for l, (px, py) in centroids.items():
        ax.scatter([px], [py], s=360, color=data[l]["color"], marker="*",
                   edgecolors="black", linewidths=1.4, zorder=6,
                   label=f"{l} (mean)")

    # Reference lines — present but SUBTLE so the Pareto frontier stands out.
    ax.axvline(0.65, color="#E65100", ls=":", lw=1.6, alpha=0.5,
               label="privacy threshold 0.65")
    ax.axhline(0.80, color="#B71C1C", ls=":", lw=1.6, alpha=0.5,
               label="utility threshold 0.80")
    ax.set_xlabel("Privacy Score", fontsize=22)
    ax.set_ylabel("Utility Score (TC@3+TA@1)/2", fontsize=22)
    ax.tick_params(axis="both", labelsize=22)
    ax.set_title("Privacy vs Utility Trade-off\n(★ = per-config mean; dashed = Pareto frontier)",
                 fontweight="bold", fontsize=24)
    # Legend outside-right with a frame; de-duplicate the faint scatter entries.
    handles, lbls = ax.get_legend_handles_labels()
    seen = {}
    for h, lb in zip(handles, lbls):
        if lb not in seen:
            seen[lb] = h
    leg = ax.legend(seen.values(), seen.keys(), loc="center left",
                    bbox_to_anchor=(1.01, 0.5), fontsize=20, frameon=True,
                    framealpha=0.95, edgecolor="black")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "privacy_utility_tradeoff.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ✓ privacy_utility_tradeoff.png")

    # ── 4. Classification Metrics ──
    metric_names = ["mAP", "accuracy", "f1"]
    metric_labels_display = ["mAP", "Accuracy", "F1-Score"]
    metric_colors = ["#1976D2", "#F57C00", "#388E3C"]
    bar_w = 0.8 / len(metric_names)
    fig, ax = plt.subplots(figsize=(14, 7))
    for j, (mname, mlabel, mcolor) in enumerate(zip(metric_names, metric_labels_display, metric_colors)):
        vals = []
        for l in labels:
            s = sc(data[l]["reports"])
            scores = [c["metrics"]["utility"][mname] for c in s if mname in c["metrics"]["utility"]]
            vals.append(np.mean(scores) if scores else 0)
        ax.bar(x + j * bar_w - (len(metric_names)-1)*bar_w/2, vals, bar_w,
               label=mlabel, color=mcolor, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.2)
    ax.set_title("Classification Metrics (mAP, Accuracy, F1)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "classification_metrics.png"), dpi=150); plt.close(fig)
    print("  ✓ classification_metrics.png")

    # ── 5. Recipe Usage (including Source Separation pre-processing) ──
    all_recipes = set()
    ver_recipes = {}
    for l in labels:
        s = sc(data[l]["reports"])
        counts = {}
        for c in s:
            # Check if source separation was used as pre-processing
            if chunk_used_ss(c):
                counts["SOURCE_SEPARATION (pre)"] = counts.get("SOURCE_SEPARATION (pre)", 0) + 1
                all_recipes.add("SOURCE_SEPARATION (pre)")
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "unknown").replace("RECIPE_", "")
                counts[name] = counts.get(name, 0) + 1
                all_recipes.add(name)
        ver_recipes[l] = counts
    recipe_colors = {"MID_BAND_ATTEN": "#2196F3", "LOWPASS_HIGHPASS_MIX": "#FF9800", "SOURCE_SEPARATION": "#4CAF50", "SOURCE_SEPARATION (pre)": "#81C784"}
    recipes = sorted(all_recipes)
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(labels))
    for recipe in recipes:
        vals = [ver_recipes[l].get(recipe, 0) for l in labels]
        ax.bar(x, vals, 0.5, bottom=bottom, label=recipe, color=recipe_colors.get(recipe, "#999"), alpha=0.85)
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Chunks"); ax.set_title("Recipe Usage (incl. Source Separation pre-processing)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "recipe_usage.png"), dpi=150); plt.close(fig)
    print("  ✓ recipe_usage.png")

    # ── 6. Trials Distribution ──
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, l in enumerate(labels):
        s = sc(data[l]["reports"])
        trials = [c.get("trials", 0) for c in s if c.get("trials", 0) > 0]
        avg = np.mean(trials) if trials else 0
        bar = ax.bar(i, avg, color=colors[i], alpha=0.85)
        ax.text(i, avg + 0.05, f"{avg:.2f}", ha="center", fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("Avg Trials"); ax.set_title("Average Trials per Chunk", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "trials_distribution.png"), dpi=150); plt.close(fig)
    print("  ✓ trials_distribution.png")

    # ── 7. Speech Ratio vs Privacy (binned-mean trend + SS stars) ──
    fig, ax = plt.subplots(figsize=(17, 10))

    # Helper: per-method binned mean (8 bins along SR axis, skip sparse bins)
    def _binned_trend(ratios, priv, n_bins=8, min_n=3):
        if not ratios:
            return [], []
        r = np.asarray(ratios); p = np.asarray(priv)
        edges = np.linspace(0, 1, n_bins + 1)
        cx, cy = [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i+1]
            mask = (r >= lo) & (r < hi if i < n_bins - 1 else r <= hi)
            if mask.sum() >= min_n:
                cx.append((lo + hi) / 2.0)
                cy.append(float(p[mask].mean()))
        return cx, cy

    # (a) Very light raw scatter in the background (keeps distribution visible).
    for l in labels:
        s = sc(data[l]["reports"])
        r = [c.get("speech_ratio", 0) for c in s]
        p = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        ax.scatter(r, p, alpha=0.06, s=12, color=data[l]["color"], marker="o", zorder=1)

    # (b) Bold per-method binned-mean trend line (one legend entry per method).
    for l in labels:
        s = sc(data[l]["reports"])
        r_all = [c.get("speech_ratio", 0) for c in s]
        p_all = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        cx, cy = _binned_trend(r_all, p_all, n_bins=8)
        if cx:
            ax.plot(cx, cy, color=data[l]["color"], lw=2.6, marker="o",
                    markersize=7, markeredgecolor="black", markeredgewidth=0.6,
                    label=l, zorder=3)

    # (c) SS-applied chunks as black-edged stars (ONE generic legend entry).
    first_star = True
    for l in labels:
        s = sc(data[l]["reports"])
        ss_only = [c for c in s if chunk_used_ss(c)]
        if ss_only:
            r = [c.get("speech_ratio", 0) for c in ss_only]
            p = [c["metrics"]["privacy"]["privacy_score"] for c in ss_only]
            star_c = data[l].get("star_color", data[l]["color"])  # pink for Rule
            ax.scatter(r, p, alpha=0.9, s=90, color=star_c,
                       marker="*", edgecolors="black", linewidths=0.7, zorder=4,
                       label="Source separation applied" if first_star else None)
            first_star = False

    # Headroom so labels/annotations sit ABOVE the data, never overlapping it.
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.35)

    # Reference lines — bold & clearly visible.
    ax.axhline(0.65, color="#E65100", ls="--", lw=3.0, alpha=0.95, zorder=2)
    # privacy label at the RIGHT end, just above the line (away from the SS line).
    ax.text(0.99, 0.67, "privacy threshold  T_p = 0.65", color="#E65100",
            fontsize=18, fontweight="bold", va="bottom", ha="right")
    ax.axvline(0.3, color="#B71C1C", ls=":", lw=3.0, alpha=0.95, zorder=2)
    # SS label vertical, to the LEFT of the line in the empty upper-left region.
    ax.text(0.275, 0.98, "SS threshold  SR = 0.3", color="#B71C1C",
            fontsize=18, fontweight="bold", va="top", ha="center", rotation=90)

    # Explanatory callouts in the TOP headroom band (y > 1.05) — clear of all data.
    ax.annotate("Rule-based: consistently high privacy (heavy blur)",
                xy=(0.62, 0.87), xytext=(0.62, 1.28),
                fontsize=20, fontweight="bold", color="#880E4F", ha="center",
                arrowprops=dict(arrowstyle="->", color="#880E4F", lw=2.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="#FCE4EC", ec="#880E4F", lw=1.5),
                zorder=8)
    ax.annotate("LLM: adaptive — trades privacy for utility (lighter blur)",
                xy=(0.80, 0.40), xytext=(0.60, 1.12),
                fontsize=20, fontweight="bold", color="#1A237E", ha="center",
                arrowprops=dict(arrowstyle="->", color="#1A237E", lw=2.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8EAF6", ec="#1A237E", lw=1.5),
                zorder=8)

    ax.set_xlabel("Speech Ratio", fontsize=22)
    ax.set_ylabel("Privacy Score", fontsize=22)
    ax.tick_params(axis="both", labelsize=22)
    ax.set_title("Speech Ratio vs Privacy Score\n"
                 "(lines = per-method binned mean; ★ = source separation applied)",
                 fontweight="bold", fontsize=24)
    # Legend OUTSIDE the axes (right) with a visible frame — never overlaps data.
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=20,
                    frameon=True, framealpha=0.97, edgecolor="black", title="Configuration")
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_fontsize(22)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "speech_ratio_vs_privacy.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ speech_ratio_vs_privacy.png")

    # ── 8. Top-K Class Consistency ──
    topk_data = {}
    for l in labels:
        s = sc(data[l]["reports"])
        consistency = []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                orig_labels = set(p["label"] for p in orig[:3])
                proc_labels = set(p["label"] for p in proc[:3])
                overlap = len(orig_labels & proc_labels)
                consistency.append(overlap / max(len(orig_labels), 1))
        if consistency:
            topk_data[l] = consistency

    if topk_data:
        tk_labels = [l for l in labels if l in topk_data]
        tk_x = np.arange(len(tk_labels))
        means = [np.mean(topk_data[l]) for l in tk_labels]
        stds = [np.std(topk_data[l]) for l in tk_labels]
        tk_colors = [data[l]["color"] for l in tk_labels]
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(tk_x, means, 0.5, yerr=stds, capsize=5, color=tk_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.4)
        ax.set_xticks(tk_x); ax.set_xticklabels(tk_labels)
        ax.set_ylabel("Consistency (0-1)"); ax.set_ylim(0, 1.4)
        ax.set_title("Top-3 Class Consistency\n(Environmental labels preserved after processing)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "topk_class_consistency.png"), dpi=150); plt.close(fig)
        print("  ✓ topk_class_consistency.png")

    # ── 9. Classification Label Change ──
    changed_counts = []
    for l in labels:
        s = sc(data[l]["reports"])
        changed, total = 0, 0
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                total += 1
                if orig[0]["label"] != proc[0]["label"]:
                    changed += 1
        changed_counts.append((l, changed, total))

    if any(t > 0 for _, _, t in changed_counts):
        fig, ax = plt.subplots(figsize=(14, 7))
        pct = [100 * ch / max(t, 1) for _, ch, t in changed_counts]
        bars = ax.bar(x, pct, w, color=colors, alpha=0.85)
        for bar, (lbl, ch, t) in zip(bars, changed_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{ch}/{t} ({bar.get_height():.1f}%)", ha="center", fontweight="bold", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("% Chunks with Changed Top-1 Label")
        ax.set_title("Classification Label Change After Processing", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "classification_label_change.png"), dpi=150); plt.close(fig)
        print("  ✓ classification_label_change.png")

    # ── 10. Classification Top Labels ──
    fig, axes = plt.subplots(1, len(labels), figsize=(6*len(labels), 6), squeeze=False)
    axes = axes[0]
    for i, l in enumerate(labels):
        ax = axes[i]
        s = sc(data[l]["reports"])
        label_counts = Counter()
        for c in s:
            top3 = c.get("classification_top3") or []
            if top3:
                label_counts[top3[0]["label"]] += 1
        if label_counts:
            top_items = label_counts.most_common(10)
            lnames = [item[0][:20] for item in top_items]
            lcounts = [item[1] for item in top_items]
            ax.barh(range(len(lnames)), lcounts, color=data[l]["color"], alpha=0.85)
            ax.set_yticks(range(len(lnames))); ax.set_yticklabels(lnames, fontsize=10)
            ax.invert_yaxis()
        ax.set_xlabel("Count"); ax.set_title(l, fontsize=13)
    fig.suptitle("Top-1 Classification Labels (Processed Audio)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "classification_top_labels.png"), dpi=150); plt.close(fig)
    print("  ✓ classification_top_labels.png")

    # ── 11. Speech Confidence Gap ──
    conf_data = {}
    for l in labels:
        s = sc(data[l]["reports"])
        gaps = []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                def find_speech_conf(preds):
                    for p in preds:
                        if "speech" in p["label"].lower():
                            return p["confidence"]
                    return 0.0
                gaps.append(find_speech_conf(orig) - find_speech_conf(proc))
        if gaps:
            conf_data[l] = gaps

    if conf_data:
        cd_labels = [l for l in labels if l in conf_data]
        cd_x = np.arange(len(cd_labels))
        means = [np.mean(conf_data[l]) for l in cd_labels]
        stds = [np.std(conf_data[l]) for l in cd_labels]
        cd_colors = [data[l]["color"] for l in cd_labels]
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(cd_x, means, 0.5, yerr=stds, capsize=5, color=cd_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                    f"{m:+.3f}±{s:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(cd_x); ax.set_xticklabels(cd_labels)
        ax.set_ylabel("Confidence Drop (positive = reduced)")
        ax.set_title("Speech Confidence Gap (Original − Processed)\n(Higher = better privacy)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "speech_confidence_gap.png"), dpi=150); plt.close(fig)
        print("  ✓ speech_confidence_gap.png")

    # ── 12. Speech Rank Drop ──
    rank_data = {}
    for l in labels:
        s = sc(data[l]["reports"])
        drops = []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                def find_speech_rank(preds):
                    for i, p in enumerate(preds):
                        if "speech" in p["label"].lower():
                            return i + 1
                    return len(preds) + 1
                drops.append(find_speech_rank(proc) - find_speech_rank(orig))
        if drops:
            rank_data[l] = drops

    if rank_data:
        rd_labels = [l for l in labels if l in rank_data]
        rd_x = np.arange(len(rd_labels))
        means = [np.mean(rank_data[l]) for l in rd_labels]
        stds = [np.std(rank_data[l]) for l in rd_labels]
        rd_colors = [data[l]["color"] for l in rd_labels]
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(rd_x, means, 0.5, yerr=stds, capsize=5, color=rd_colors, alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + s + 0.1,
                    f"{m:+.2f}±{s:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(rd_x); ax.set_xticklabels(rd_labels)
        ax.set_ylabel("Rank Drop (positive = dropped)")
        ax.set_title("Speech Rank Drop After Processing\n(Higher = better privacy)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "speech_rank_drop.png"), dpi=150); plt.close(fig)
        print("  ✓ speech_rank_drop.png")

    # ── 13. No-Speech Analysis [v2: semantic preserve] ──
    ns_labels, ns_means, ns_stds, ns_counts = [], [], [], []
    for l in labels:
        ns = no_sc(data[l]["reports"])
        if ns:
            scores = preserve_list(ns)
            if scores:
                ns_labels.append(l)
                ns_means.append(np.mean(scores))
                ns_stds.append(np.std(scores))
                ns_counts.append(len(scores))

    if ns_labels:
        ns_x = np.arange(len(ns_labels))
        ns_colors = [data[l]["color"] for l in ns_labels]
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(ns_x, ns_means, 0.5, yerr=ns_stds, capsize=5, color=ns_colors, alpha=0.85)
        for bar, m, s, n in zip(bars, ns_means, ns_stds, ns_counts):
            ax.text(bar.get_x() + bar.get_width()/2, min(bar.get_height() + s + 0.005, 1.05),
                    f"{m:.4f}±{s:.4f} (n={n})", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(ns_x); ax.set_xticklabels(ns_labels)
        ax.set_ylabel("Utility Score"); ax.set_ylim(0.6, 1.1)
        ax.set_title("Utility Score — No-Speech Chunks (Bypass)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "no_speech_utility.png"), dpi=150); plt.close(fig)
        print("  ✓ no_speech_utility.png")

    # ── 14. Source Separation Analysis ──
    # Identify chunks that used source separation
    ss_analysis = {}
    for l in labels:
        s = sc(data[l]["reports"])
        ss_chunks = [c for c in s if chunk_used_ss(c)]
        non_ss_chunks = [c for c in s if not chunk_used_ss(c)]
        ss_analysis[l] = {"ss": ss_chunks, "non_ss": non_ss_chunks, "total": len(s)}

    # 14a. SS usage count
    fig, ax = plt.subplots(figsize=(14, 7))
    ss_counts = [len(ss_analysis[l]["ss"]) for l in labels]
    non_ss_counts = [len(ss_analysis[l]["non_ss"]) for l in labels]
    bar_w2 = 0.35
    b1 = ax.bar(x - bar_w2/2, non_ss_counts, bar_w2, label="Without SS", color="#90CAF9", alpha=0.85)
    b2 = ax.bar(x + bar_w2/2, ss_counts, bar_w2, label="With SS", color="#4CAF50", alpha=0.85)
    for bar, val in zip(b1, non_ss_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha="center", fontweight="bold", fontsize=11)
    for bar, val in zip(b2, ss_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha="center", fontweight="bold", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Chunks")
    ax.set_title("Source Separation Usage (Speech Chunks)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "source_sep_usage.png"), dpi=150); plt.close(fig)
    print("  ✓ source_sep_usage.png")

    # 14b. Privacy score: SS vs non-SS
    fig, ax = plt.subplots(figsize=(14, 7))
    ss_priv = []
    non_ss_priv = []
    plot_labels = []
    for l in labels:
        ss = ss_analysis[l]["ss"]
        non_ss = ss_analysis[l]["non_ss"]
        if ss or non_ss:
            plot_labels.append(l)
            ss_priv.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in ss]) if ss else 0)
            non_ss_priv.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in non_ss]) if non_ss else 0)
    if plot_labels:
        px = np.arange(len(plot_labels))
        b1 = ax.bar(px - bar_w2/2, non_ss_priv, bar_w2, label="Without SS", color="#90CAF9", alpha=0.85)
        b2 = ax.bar(px + bar_w2/2, ss_priv, bar_w2, label="With SS", color="#4CAF50", alpha=0.85)
        for bar, m in zip(b1, non_ss_priv):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{m:.3f}", ha="center", fontweight="bold", fontsize=11)
        for bar, m in zip(b2, ss_priv):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{m:.3f}", ha="center", fontweight="bold", fontsize=11)
        ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(px); ax.set_xticklabels(plot_labels)
        ax.set_ylabel("Privacy Score"); ax.set_ylim(0, 1.2)
        ax.set_title("Privacy Score: With vs Without Source Separation", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "source_sep_privacy.png"), dpi=150); plt.close(fig)
        print("  ✓ source_sep_privacy.png")

    # 14c. All metrics comparison: SS vs non-SS  [v2: semantic preserve for "preserve_score"]
    metric_keys = [("privacy_score", "Privacy"), ("preserve_score", "Utility"), ("mAP", "mAP"), ("accuracy", "Accuracy"), ("f1", "F1")]
    rows_ss = []
    for l in labels:
        ss = ss_analysis[l]["ss"]
        non_ss = ss_analysis[l]["non_ss"]
        row = [l, str(len(ss)), str(len(non_ss))]
        for key, _ in metric_keys:
            if key == "privacy_score":
                ss_val = np.mean([c["metrics"]["privacy"][key] for c in ss]) if ss else 0
                non_val = np.mean([c["metrics"]["privacy"][key] for c in non_ss]) if non_ss else 0
            elif key == "preserve_score":
                ss_scores = preserve_list(ss)
                non_scores = preserve_list(non_ss)
                ss_val = np.mean(ss_scores) if ss_scores else 0
                non_val = np.mean(non_scores) if non_scores else 0
            else:
                ss_val = np.mean([c["metrics"]["utility"][key] for c in ss if key in c["metrics"]["utility"]]) if ss else 0
                non_val = np.mean([c["metrics"]["utility"][key] for c in non_ss if key in c["metrics"]["utility"]]) if non_ss else 0
            row.append(f"{ss_val:.4f}" if ss else "N/A")
            row.append(f"{non_val:.4f}" if non_ss else "N/A")
        rows_ss.append(row)

    cols_ss = ["Version", "SS\nChunks", "Non-SS\nChunks"]
    for _, name in metric_keys:
        cols_ss.extend([f"{name}\n(SS)", f"{name}\n(No SS)"])

    fig, ax = plt.subplots(figsize=(24, 2 + 0.6 * len(rows_ss)))
    ax.axis("off")
    table = ax.table(cellText=rows_ss, colLabels=cols_ss, loc="center", cellLoc="center", colColours=["#E8F5E9"] * len(cols_ss))
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.8)
    ax.set_title("Source Separation Impact Summary", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "source_sep_summary.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ✓ source_sep_summary.png")

    # ── 15. Speech Density Filtered Analysis [v2: semantic preserve] ──
    fig, ax = plt.subplots(figsize=(14, 7))
    metric_keys_density = [("privacy_score", "privacy", "Privacy Score"), ("preserve_score", "utility", "Utility Score"), ("accuracy", "utility", "Accuracy")]
    bar_wd = 0.8 / (len(labels) * 2)
    group_w = 0.8 / len(metric_keys_density)
    for j, (key, section, mlabel) in enumerate(metric_keys_density):
        all_vals, high_vals = [], []
        for l in labels:
            s = sc(data[l]["reports"])
            high = [c for c in s if c.get("speech_ratio", 0) >= 0.5]
            if section == "privacy":
                all_vals.append(np.mean([c["metrics"][section][key] for c in s]) if s else 0)
                high_vals.append(np.mean([c["metrics"][section][key] for c in high]) if high else 0)
            elif key == "preserve_score":
                all_scores = preserve_list(s)
                high_scores = preserve_list(high)
                all_vals.append(np.mean(all_scores) if all_scores else 0)
                high_vals.append(np.mean(high_scores) if high_scores else 0)
            else:
                all_vals.append(np.mean([c["metrics"][section][key] for c in s if key in c["metrics"][section]]) if s else 0)
                high_vals.append(np.mean([c["metrics"][section][key] for c in high if key in c["metrics"][section]]) if high else 0)
        gx = j * (len(labels) + 1)
        for i, (av, hv, l) in enumerate(zip(all_vals, high_vals, labels)):
            ax.bar(gx + i*0.4, av, 0.18, color=data[l]["color"], alpha=0.5)
            ax.bar(gx + i*0.4 + 0.18, hv, 0.18, color=data[l]["color"], alpha=1.0)
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="grey", alpha=0.5, label="All Chunks"),
                       Patch(facecolor="grey", alpha=1.0, label="High Density (SR≥0.5)")]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0)
    xtick_pos = [j * (len(labels) + 1) + len(labels)*0.2 for j in range(len(metric_keys_density))]
    ax.set_xticks(xtick_pos); ax.set_xticklabels([m[2] for m in metric_keys_density])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.2)
    ax.set_title("Impact of Speech Density: All vs High-Density Chunks (SR≥0.5)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "speech_density_analysis.png"), dpi=150); plt.close(fig)
    print("  ✓ speech_density_analysis.png")

    # ── 16. WER vs Privacy Score (Scatter + Regression) ──
    fig, ax = plt.subplots(figsize=(12, 8))
    for l in labels:
        s = sc(data[l]["reports"])
        wer = [c["metrics"]["privacy"]["wer"] for c in s]
        priv = [c["metrics"]["privacy"]["privacy_score"] for c in s]
        ax.scatter(wer, priv, alpha=0.4, s=25, label=l, color=data[l]["color"])
    # Overall regression line
    all_wer, all_priv = [], []
    for l in labels:
        s = sc(data[l]["reports"])
        all_wer.extend([c["metrics"]["privacy"]["wer"] for c in s])
        all_priv.extend([c["metrics"]["privacy"]["privacy_score"] for c in s])
    if all_wer:
        z = np.polyfit(all_wer, all_priv, 1)
        p = np.poly1d(z)
        wx = np.linspace(0, 1, 100)
        ax.plot(wx, p(wx), "r--", lw=2, alpha=0.7, label=f"Regression (r={np.corrcoef(all_wer, all_priv)[0,1]:.3f})")
    ax.set_xlabel("WER (Word Error Rate)"); ax.set_ylabel("Privacy Score")
    ax.set_title("WER vs Privacy Score\n(Higher WER = speech more obscured)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "wer_vs_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ wer_vs_privacy.png")

    # ── 17. WER Distribution by Recipe Level ──
    fig, ax = plt.subplots(figsize=(14, 7))
    for l in labels:
        s = sc(data[l]["reports"])
        wer_vals = [c["metrics"]["privacy"]["wer"] for c in s]
        if wer_vals:
            ax.hist(wer_vals, bins=20, alpha=0.4, label=l, color=data[l]["color"], edgecolor="white")
    ax.set_xlabel("WER"); ax.set_ylabel("Count")
    ax.set_title("WER Distribution (Content Privacy)\n(WER=1.0 means completely unintelligible)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "wer_distribution.png"), dpi=150); plt.close(fig)
    print("  ✓ wer_distribution.png")

    # ── 18. Utility by Sound Class (Box Plot) [v2: semantic preserve] ──
    # Use classification_top3_original to group by sound type
    class_utility = {}
    for l in labels:
        s = sc(data[l]["reports"])
        for c in s:
            top3_orig = c.get("classification_top3_original", [])
            ps = semantic_preserve_score(c)
            if top3_orig and ps is not None:
                sound_class = top3_orig[0]["label"]
                class_utility.setdefault(sound_class, []).append(ps)

    if class_utility:
        # Top 10 most common classes
        sorted_classes = sorted(class_utility.items(), key=lambda x: -len(x[1]))[:10]
        class_names = [c[0][:20] for c in sorted_classes]
        class_scores = [c[1] for c in sorted_classes]
        fig, ax = plt.subplots(figsize=(16, 7))
        try:
            bp = ax.boxplot(class_scores, tick_labels=class_names, patch_artist=True)
        except TypeError:  # older matplotlib still uses labels=
            bp = ax.boxplot(class_scores, labels=class_names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#90CAF9"); patch.set_alpha(0.7)
        ax.axhline(0.80, color="red", ls="--", lw=1, alpha=0.7, label="Threshold (0.80)")
        ax.set_ylabel("Utility Score")
        ax.set_title("Utility Score by Original Sound Class\n(Environmental sound preservation across categories)", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "utility_by_sound_class.png"), dpi=150); plt.close(fig)
        print("  ✓ utility_by_sound_class.png")

    # ── 19. Per-Class Analysis Table [v2: semantic preserve] ──
    # Privacy and Utility broken down by original sound class
    class_metrics = {}
    for l in labels:
        s = sc(data[l]["reports"])
        for c in s:
            top3_orig = c.get("classification_top3_original", [])
            ps = semantic_preserve_score(c)
            if top3_orig and ps is not None:
                cls = top3_orig[0]["label"]
                class_metrics.setdefault(cls, {}).setdefault(l, []).append({
                    "privacy": c["metrics"]["privacy"]["privacy_score"],
                    "utility": ps,
                    "wer": c["metrics"]["privacy"]["wer"],
                })

    if class_metrics:
        # Top 8 classes by count
        top_classes = sorted(class_metrics.items(), key=lambda x: -sum(len(v) for v in x[1].values()))[:8]
        cols_pc = ["Class", "n"] + [f"{l}\nPrivacy" for l in labels] + [f"{l}\nUtility" for l in labels]
        rows_pc = []
        for cls, ver_data in top_classes:
            row = [cls[:18], str(sum(len(v) for v in ver_data.values()))]
            for l in labels:
                chunks = ver_data.get(l, [])
                row.append(f"{np.mean([c['privacy'] for c in chunks]):.3f}" if chunks else "—")
            for l in labels:
                chunks = ver_data.get(l, [])
                row.append(f"{np.mean([c['utility'] for c in chunks]):.3f}" if chunks else "—")
            rows_pc.append(row)

        fig, ax = plt.subplots(figsize=(24, 2 + 0.5 * len(rows_pc)))
        ax.axis("off")
        table = ax.table(cellText=rows_pc, colLabels=cols_pc, loc="center", cellLoc="center", colColours=["#E8F5E9"] * len(cols_pc))
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.7)
        ax.set_title("Per-Class Privacy & Utility Analysis", fontweight="bold", pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "per_class_analysis.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ✓ per_class_analysis.png")

    # ── 20. Chunk-Level Paired Comparison (Rule vs LLM-Mem-SS) ──
    rule_key = "Rule-NoMem-SS"
    llm_key = "LLM-H4.5-Mem-SS"
    if rule_key in data and llm_key in data:
        rule_reports = data[rule_key]["reports"]
        llm_reports = data[llm_key]["reports"]
        # Match by source_id
        rule_by_src = {r["source_id"]: r for r in rule_reports}
        llm_by_src = {r["source_id"]: r for r in llm_reports}
        common_srcs = set(rule_by_src.keys()) & set(llm_by_src.keys())

        rule_priv, llm_priv = [], []
        for src in common_srcs:
            rc = sc([rule_by_src[src]])
            lc = sc([llm_by_src[src]])
            if rc and lc:
                rule_priv.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in rc]))
                llm_priv.append(np.mean([c["metrics"]["privacy"]["privacy_score"] for c in lc]))

        if rule_priv:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(rule_priv, llm_priv, alpha=0.4, s=30, color="#4CAF50")
            ax.plot([0, 1], [0, 1], "r--", lw=1, alpha=0.7, label="Equal line")
            ax.set_xlabel(f"Privacy Score ({rule_key})"); ax.set_ylabel(f"Privacy Score ({llm_key})")
            ax.set_title(f"Chunk-Level Paired Comparison\n{llm_key} vs {rule_key}", fontweight="bold")
            # Count wins
            llm_wins = sum(1 for r, l in zip(rule_priv, llm_priv) if l > r)
            rule_wins = sum(1 for r, l in zip(rule_priv, llm_priv) if r > l)
            ties = len(rule_priv) - llm_wins - rule_wins
            ax.text(0.05, 0.95, f"LLM wins: {llm_wins}\nRule wins: {rule_wins}\nTies: {ties}\nn={len(rule_priv)}",
                    transform=ax.transAxes, va="top", fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(alpha=0.3)
            fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "paired_comparison_rule_vs_llm.png"), dpi=150); plt.close(fig)
            print("  ✓ paired_comparison_rule_vs_llm.png")

    # ── 21. WER-Only Privacy Ranking ──
    # Compare methods using WER alone as privacy metric
    fig, ax = plt.subplots(figsize=(14, 7))
    wer_means = []
    for l in labels:
        s = sc(data[l]["reports"])
        wer_vals = [c["metrics"]["privacy"]["wer"] for c in s]
        wer_means.append(np.mean(wer_vals) if wer_vals else 0)
    bars = ax.bar(x, wer_means, w, color=colors, alpha=0.85)
    for bar, m in zip(bars, wer_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Average WER"); ax.set_ylim(0, 1.2)
    ax.set_title("WER-Only Privacy Ranking\n(WER=1.0 = completely unintelligible speech)", fontweight="bold")
    ax.axhline(0.65, color="orange", ls="--", lw=1, alpha=0.7, label="Privacy threshold (WER≥0.65)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "wer_only_ranking.png"), dpi=150); plt.close(fig)
    print("  ✓ wer_only_ranking.png")

    # ── 22. Semantic Preservation Metrics (TC@3, TA@1, PreserveScore) ──
    sem_data = {}
    for l in labels:
        s = sc(data[l]["reports"])
        tc3_scores, ta1_scores = [], []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if not orig or not proc:
                continue
            # TC@3: Top-3 Consistency
            set_a = set(p["label"] for p in orig[:3])
            set_b = set(p["label"] for p in proc[:3])
            tc3 = len(set_a & set_b) / 3.0
            tc3_scores.append(tc3)
            # TA@1: Top-1 Agreement
            ta1 = 1.0 if orig[0]["label"] == proc[0]["label"] else 0.0
            ta1_scores.append(ta1)
        if tc3_scores:
            preserve_scores = [(tc + ta) / 2.0 for tc, ta in zip(tc3_scores, ta1_scores)]
            sem_data[l] = {"tc3": tc3_scores, "ta1": ta1_scores, "preserve": preserve_scores}

    if sem_data:
        sem_labels = [l for l in labels if l in sem_data]
        sem_x = np.arange(len(sem_labels))
        sem_colors = [data[l]["color"] for l in sem_labels]

        # 22a. Grouped bar: TC@3, TA@1, PreserveScore
        metrics_sem = [("tc3", "TC@3"), ("ta1", "TA@1"), ("preserve", "Semantic PreserveScore")]
        bar_ws = 0.8 / len(metrics_sem)
        fig, ax = plt.subplots(figsize=(26, 11))
        for j, (key, mlabel) in enumerate(metrics_sem):
            means = [np.mean(sem_data[l][key]) for l in sem_labels]
            # Drawn error bar = ±1 SD (chunk spread). 95% CI of the mean =
            # 1.96*SD/sqrt(n) is tiny (n in the thousands) but still reported.
            sds, cis = [], []
            for l in sem_labels:
                arr = np.asarray(sem_data[l][key], dtype=float)
                n = max(len(arr), 1)
                sd = arr.std(ddof=1) if n > 1 else 0.0
                sds.append(sd)
                cis.append(1.96 * sd / np.sqrt(n) if n > 1 else 0.0)
            bars = ax.bar(sem_x + j * bar_ws - (len(metrics_sem)-1)*bar_ws/2, means, bar_ws,
                          yerr=sds, label=mlabel, alpha=0.85,
                          error_kw=dict(ecolor="black", elinewidth=1.6, capsize=5, capthick=1.6))
            # Compact 3-line label above each error-bar cap: mean / ±SD / 95%CI.
            for bar, m, sd, ci in zip(bars, means, sds, cis):
                ax.text(bar.get_x() + bar.get_width()/2,
                        min(bar.get_height() + sd + 0.02, 1.55),
                        f"{m:.2f}\n±{sd:.2f}\nCI±{ci:.3f}", ha="center", va="bottom",
                        fontweight="bold", fontsize=24, linespacing=1.15)
        ax.set_xticks(sem_x); ax.set_xticklabels(sem_labels)
        ax.set_ylabel("Score"); ax.set_ylim(0, 2.05)
        ax.set_title("Semantic Preservation Metrics  (error bars = ±1 SD; CI = 95% CI of mean)\n"
                     "TC@3 (Top-3 Consistency) | TA@1 (Top-1 Agreement) | Utility Score",
                     fontweight="bold", pad=24)
        # Legend well below the x-axis labels with a frame — never overlaps bars.
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(metrics_sem),
                        borderaxespad=0, frameon=True, framealpha=0.95, edgecolor="black")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.14, 1, 1])
        fig.savefig(os.path.join(OUT_DIR, "semantic_preservation.png"), dpi=150,
                    bbox_inches="tight"); plt.close(fig)
        print("  ✓ semantic_preservation.png")

        # 22b. TC@3 distribution (histogram per version)
        fig, ax = plt.subplots(figsize=(14, 7))
        for l in sem_labels:
            ax.hist(sem_data[l]["tc3"], bins=[0, 0.33, 0.67, 1.01], alpha=0.5,
                    label=f"{l} (μ={np.mean(sem_data[l]['tc3']):.3f})", color=data[l]["color"], edgecolor="white")
        ax.set_xlabel("TC@3 Score"); ax.set_ylabel("Count")
        ax.set_xticks([0, 0.33, 0.67, 1.0]); ax.set_xticklabels(["0/3", "1/3", "2/3", "3/3"])
        ax.set_title("Top-3 Consistency Distribution\n(How many environmental labels preserved)", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "tc3_distribution.png"), dpi=150); plt.close(fig)
        print("  ✓ tc3_distribution.png")

        # 22c. TA@1 rate (bar chart — % of chunks where top-1 class unchanged)
        fig, ax = plt.subplots(figsize=(14, 7))
        ta1_rates = [100 * np.mean(sem_data[l]["ta1"]) for l in sem_labels]
        bars = ax.bar(sem_x, ta1_rates, w, color=sem_colors, alpha=0.85)
        for bar, rate in zip(bars, ta1_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12)
        ax.set_xticks(sem_x); ax.set_xticklabels(sem_labels)
        ax.set_ylabel("TA@1 Rate (%)"); ax.set_ylim(0, 110)
        ax.set_title("Top-1 Agreement Rate\n(% chunks where dominant sound class unchanged after processing)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "ta1_rate.png"), dpi=150); plt.close(fig)
        print("  ✓ ta1_rate.png")

        # 22d. Summary table for semantic metrics
        cols_sem = ["Version", "n", "TC@3\nMean±σ", "TA@1\nRate", "PreserveScore\nMean±σ"]
        rows_sem = []
        for l in sem_labels:
            n = len(sem_data[l]["tc3"])
            rows_sem.append([
                l, str(n),
                f"{np.mean(sem_data[l]['tc3']):.3f}±{np.std(sem_data[l]['tc3']):.3f}",
                f"{100*np.mean(sem_data[l]['ta1']):.1f}%",
                f"{np.mean(sem_data[l]['preserve']):.3f}±{np.std(sem_data[l]['preserve']):.3f}",
            ])
        fig, ax = plt.subplots(figsize=(16, 2 + 0.5 * len(rows_sem)))
        ax.axis("off")
        table = ax.table(cellText=rows_sem, colLabels=cols_sem, loc="center", cellLoc="center", colColours=["#E3F2FD"] * len(cols_sem))
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
        ax.set_title("Semantic Preservation Summary", fontweight="bold", pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "semantic_preservation_summary.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ✓ semantic_preservation_summary.png")

    # ── 23. Average Speaker Privacy ──
    means_sp = [np.mean([c["metrics"]["privacy"]["speaker_privacy"] for c in sc(data[l]["reports"])]) for l in labels]
    stds_sp = [np.std([c["metrics"]["privacy"]["speaker_privacy"] for c in sc(data[l]["reports"])]) for l in labels]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(x, means_sp, w, yerr=stds_sp, capsize=5, color=colors, alpha=0.85)
    for bar, m, s in zip(bars, means_sp, stds_sp):
        annotate_bar(ax, bar, f"{m:.3f}±{s:.3f}", 1.3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Speaker Privacy Score"); ax.set_ylim(0, 1.3)
    ax.set_title("Average Speaker Privacy (±σ)\n(1 - cosine similarity of speaker embeddings)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(os.path.join(OUT_DIR, "avg_speaker_privacy.png"), dpi=150); plt.close(fig)
    print("  ✓ avg_speaker_privacy.png")

    # ── 24. Summary Table [v2: semantic preserve + TC@3 + TA@1] ──
    cols = ["Version", "Speech\nChunks", "Avg Privacy", "Avg Utility\n(TC@3+TA@1)/2", "TC@3\nMean", "TA@1\nRate", "Avg Trials", "Recipes"]
    rows = []
    for l in labels:
        s = sc(data[l]["reports"])
        recipes = {}
        for c in s:
            r = c.get("recipe_applied", {})
            if r:
                name = r.get("recipe_name", "?").replace("RECIPE_", "")
                recipes[name] = recipes.get(name, 0) + 1
        # Compute semantic metrics
        tc3_vals, ta1_vals = [], []
        for c in s:
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            if orig and proc:
                set_a = set(p["label"] for p in orig[:3])
                set_b = set(p["label"] for p in proc[:3])
                tc3_vals.append(len(set_a & set_b) / 3.0)
                ta1_vals.append(1.0 if orig[0]["label"] == proc[0]["label"] else 0.0)
        ps_list = preserve_list(s)
        rows.append([
            l,
            str(len(s)),
            f"{np.mean([c['metrics']['privacy']['privacy_score'] for c in s]):.4f}" if s else "N/A",
            f"{np.mean(ps_list):.4f}" if ps_list else "N/A",
            f"{np.mean(tc3_vals):.4f}" if tc3_vals else "N/A",
            f"{100*np.mean(ta1_vals):.1f}%" if ta1_vals else "N/A",
            f"{np.mean([c.get('trials', 0) for c in s]):.2f}" if s else "N/A",
            ", ".join(f"{k}:{v}" for k, v in sorted(recipes.items())),
        ])
    fig, ax = plt.subplots(figsize=(20, 2 + 0.6 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colColours=["#E3F2FD"] * len(cols))
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
    ax.set_title("Summary [Semantic Preserve Metric]", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_table.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ✓ summary_table.png")

    # ─────────────────────────────────────────────────────────────────────
    # [v2 PATCH] PAPER-READY OUTPUTS
    #   1. Multi-threshold acceptance grid (Table III in paper)
    #   2. CSV exports of per-chunk metrics + summary
    #   3. Console print of all paper tables in copy-paste form
    # ─────────────────────────────────────────────────────────────────────

    import csv

    # Threshold grids for paper
    T_P_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]
    T_U_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    T_S_DEFAULT = 0.50

    # Compute per-chunk paper metrics
    paper_chunks = {}  # cfg -> list of dicts
    for l in labels:
        s = sc(data[l]["reports"])
        rows_pchunk = []
        for c in s:
            ps = semantic_preserve_score(c)
            if ps is None:
                continue
            priv = c["metrics"]["privacy"].get("privacy_score")
            spk = c["metrics"]["privacy"].get("speaker_privacy")
            wer = c["metrics"]["privacy"].get("wer")
            cer = c["metrics"]["privacy"].get("cer")
            cp = c["metrics"]["privacy"].get("content_privacy")
            orig = c.get("classification_top3_original", [])
            proc = c.get("classification_top3", [])
            set_a = set(p["label"] for p in orig[:3])
            set_b = set(p["label"] for p in proc[:3])
            tc3 = len(set_a & set_b) / 3.0
            ta1 = 1.0 if orig[0]["label"] == proc[0]["label"] else 0.0
            rows_pchunk.append({
                "configuration": l,
                "source_id": c.get("source_id", ""),
                "chunk_id": c.get("chunk_id", ""),
                "speech_ratio": c.get("speech_ratio"),
                "vad_confidence": c.get("vad_confidence"),
                "wer": wer, "cer": cer, "content_privacy": cp,
                "speaker_privacy": spk, "privacy_score": priv,
                "tc3": tc3, "ta1": ta1, "preserve_new": ps,
                "trials_used": c.get("trials"),
                "used_ss": chunk_used_ss(c),
                "recipe": (c.get("recipe_applied") or {}).get("recipe_name"),
            })
        paper_chunks[l] = rows_pchunk

    # ── A. Per-chunk CSV ──
    pchunk_path = os.path.join(OUT_DIR, "per_chunk_metrics_paper.csv")
    with open(pchunk_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["configuration","source_id","chunk_id","speech_ratio","vad_confidence",
                       "wer","cer","content_privacy","speaker_privacy","privacy_score",
                       "tc3","ta1","preserve_new","trials_used","used_ss","recipe"])
        for l in labels:
            for r in paper_chunks[l]:
                wcsv.writerow([r["configuration"], r["source_id"], r["chunk_id"],
                               r["speech_ratio"], r["vad_confidence"],
                               r["wer"], r["cer"], r["content_privacy"],
                               r["speaker_privacy"], r["privacy_score"],
                               r["tc3"], r["ta1"], r["preserve_new"],
                               r["trials_used"], r["used_ss"], r["recipe"]])
    print(f"  ✓ per_chunk_metrics_paper.csv ({sum(len(paper_chunks[l]) for l in labels)} rows)")

    # ── B. Summary CSV ──
    summary_path = os.path.join(OUT_DIR, "summary_table_paper.csv")
    with open(summary_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Configuration","n_chunks",
                       "PrivacyScore_mean","PrivacyScore_std",
                       "PreserveScore_mean","PreserveScore_std",
                       "TC@3_mean","TC@3_std",
                       "TA@1_rate",
                       "SpeakerPrivacy_mean","SpeakerPrivacy_std",
                       "WER_mean","CER_mean",
                       "AvgTrials"])
        for l in labels:
            rs = paper_chunks[l]
            if not rs:
                continue
            def _mean(xs):
                xs = [x for x in xs if x is not None]
                return sum(xs)/len(xs) if xs else 0.0
            def _std(xs):
                xs = [x for x in xs if x is not None]
                if len(xs) < 2: return 0.0
                m = _mean(xs)
                return (sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5
            wcsv.writerow([
                l, len(rs),
                f"{_mean([r['privacy_score'] for r in rs]):.4f}", f"{_std([r['privacy_score'] for r in rs]):.4f}",
                f"{_mean([r['preserve_new'] for r in rs]):.4f}", f"{_std([r['preserve_new'] for r in rs]):.4f}",
                f"{_mean([r['tc3'] for r in rs]):.4f}", f"{_std([r['tc3'] for r in rs]):.4f}",
                f"{_mean([r['ta1'] for r in rs]):.4f}",
                f"{_mean([r['speaker_privacy'] for r in rs]):.4f}", f"{_std([r['speaker_privacy'] for r in rs]):.4f}",
                f"{_mean([r['wer'] for r in rs]):.4f}",
                f"{_mean([r['cer'] for r in rs]):.4f}",
                f"{_mean([r['trials_used'] for r in rs]):.2f}",
            ])
    print(f"  ✓ summary_table_paper.csv")

    # ── C. Multi-threshold acceptance grid CSV ──
    grid_path = os.path.join(OUT_DIR, "multi_threshold_acceptance_paper.csv")
    with open(grid_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Configuration","T_p","T_u","T_s",
                       "AcceptanceRate","PrivacyPass","UtilityPass","SpeakerPass","n_chunks"])
        for l in labels:
            rs = paper_chunks[l]
            n = len(rs)
            if n == 0:
                continue
            for tp in T_P_GRID:
                for tu in T_U_GRID:
                    p_pass = sum(1 for r in rs if r["privacy_score"] is not None and r["privacy_score"] >= tp) / n
                    u_pass = sum(1 for r in rs if r["preserve_new"] is not None and r["preserve_new"] >= tu) / n
                    s_pass = sum(1 for r in rs if r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_DEFAULT) / n
                    accept = sum(1 for r in rs
                                 if r["privacy_score"] is not None and r["privacy_score"] >= tp
                                 and r["preserve_new"] is not None and r["preserve_new"] >= tu
                                 and r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_DEFAULT) / n
                    wcsv.writerow([l, f"{tp:.2f}", f"{tu:.2f}", f"{T_S_DEFAULT:.2f}",
                                   f"{accept:.4f}", f"{p_pass:.4f}", f"{u_pass:.4f}", f"{s_pass:.4f}", n])
    print(f"  ✓ multi_threshold_acceptance_paper.csv")

    # ── C2. Acceptance Rate Bar Chart at design spec (T_p=0.65, T_u=0.80, T_s=0.50) ──
    T_P_FIXED, T_U_FIXED, T_S_FIXED = 0.65, 0.80, 0.50
    ar_p, ar_u, ar_s, ar_joint = [], [], [], []
    ar_labels = []
    for l in labels:
        rs = paper_chunks[l]
        n = len(rs)
        if n == 0:
            continue
        p_pass = sum(1 for r in rs if r["privacy_score"] is not None and r["privacy_score"] >= T_P_FIXED) / n
        u_pass = sum(1 for r in rs if r["preserve_new"] is not None and r["preserve_new"] >= T_U_FIXED) / n
        s_pass = sum(1 for r in rs if r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_FIXED) / n
        joint = sum(1 for r in rs
                    if r["privacy_score"] is not None and r["privacy_score"] >= T_P_FIXED
                    and r["preserve_new"] is not None and r["preserve_new"] >= T_U_FIXED
                    and r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_FIXED) / n
        ar_labels.append(l)
        ar_p.append(p_pass); ar_u.append(u_pass); ar_s.append(s_pass); ar_joint.append(joint)

    if ar_labels:
        fig, ax = plt.subplots(figsize=(16, 8))
        ar_x = np.arange(len(ar_labels))
        bw = 0.20
        b1 = ax.bar(ar_x - 1.5*bw, ar_p,     bw, label=f"P-pass (≥{T_P_FIXED})", color="#42A5F5", alpha=0.9)
        b2 = ax.bar(ar_x - 0.5*bw, ar_u,     bw, label=f"U-pass (≥{T_U_FIXED})", color="#66BB6A", alpha=0.9)
        b3 = ax.bar(ar_x + 0.5*bw, ar_s,     bw, label=f"S-pass (≥{T_S_FIXED})", color="#FFA726", alpha=0.9)
        b4 = ax.bar(ar_x + 1.5*bw, ar_joint, bw, label="Joint Acceptance",       color="#AB47BC", alpha=0.95)
        for bars in (b1, b2, b3, b4):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h*100:.1f}%", ha="center", va="bottom", fontsize=18)
        ax.set_xticks(ar_x)
        ax.set_xticklabels(ar_labels, rotation=15, ha="right")
        ax.set_ylabel("Pass Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_title(f"Acceptance Rate  "
                     f"(T_p={T_P_FIXED}, T_u={T_U_FIXED}, T_s={T_S_FIXED})",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "acceptance_rate.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ acceptance_rate.png")

    # ── D. Console summary (paper-ready text tables) ──
    print("\n" + "=" * 90)
    print("PAPER TABLE II — Summary (semantic preserve metric)")
    print("=" * 90)
    print(f"{'Configuration':<22} {'Privacy':>10} {'Preserve':>10} {'TC@3':>8} {'TA@1':>8} {'SpkPriv':>8} {'WER':>8}")
    print("-" * 90)
    for l in labels:
        rs = paper_chunks[l]
        if not rs:
            continue
        def _mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs)/len(xs) if xs else 0.0
        print(f"{l:<22} "
              f"{_mean([r['privacy_score'] for r in rs]):>10.4f} "
              f"{_mean([r['preserve_new'] for r in rs]):>10.4f} "
              f"{_mean([r['tc3'] for r in rs]):>8.4f} "
              f"{_mean([r['ta1'] for r in rs]):>8.2%} "
              f"{_mean([r['speaker_privacy'] for r in rs]):>8.4f} "
              f"{_mean([r['wer'] for r in rs]):>8.4f}")

    print("\n" + "=" * 90)
    print("PAPER TABLE III — Acceptance rates (T_s = 0.50)")
    print("=" * 90)
    key_cells = [(0.50,0.55),(0.55,0.65),(0.60,0.55),(0.60,0.70),(0.65,0.55),(0.65,0.70),(0.65,0.80),(0.70,0.80)]
    print(f"{'Configuration':<22} | " + " | ".join(f"({tp:.2f},{tu:.2f})" for tp,tu in key_cells))
    print("-" * (22 + len(key_cells) * 13))
    for l in labels:
        rs = paper_chunks[l]
        n = len(rs)
        if n == 0:
            continue
        line = f"{l:<22} | "
        for tp, tu in key_cells:
            accept = sum(1 for r in rs
                         if r["privacy_score"] is not None and r["privacy_score"] >= tp
                         and r["preserve_new"] is not None and r["preserve_new"] >= tu
                         and r["speaker_privacy"] is not None and r["speaker_privacy"] >= T_S_DEFAULT) / n
            line += f"  {accept:>5.1%}    | "
        print(line)

    print("\n" + "=" * 90)
    print("MARGINAL PASS RATES at design spec (T_p=0.65, T_u=0.80, T_s=0.50)")
    print("=" * 90)
    for l in labels:
        rs = paper_chunks[l]
        n = len(rs)
        if n == 0:
            continue
        p_pass = sum(1 for r in rs if r["privacy_score"] is not None and r["privacy_score"] >= 0.65) / n
        u_pass = sum(1 for r in rs if r["preserve_new"] is not None and r["preserve_new"] >= 0.80) / n
        s_pass = sum(1 for r in rs if r["speaker_privacy"] is not None and r["speaker_privacy"] >= 0.50) / n
        print(f"{l:<22}  P_pass={p_pass:>6.2%}  U_pass={u_pass:>6.2%}  S_pass={s_pass:>6.2%}")

    print("\n" + "=" * 90)
    print("FILES TO HAND TO PAPER")
    print("=" * 90)
    print(f"  Numbers : {summary_path}")
    print(f"            {grid_path}")
    print(f"            {pchunk_path}")
    print(f"  Figures : {OUT_DIR}/privacy_utility_tradeoff.png")
    print(f"            {OUT_DIR}/semantic_preservation.png")
    print(f"            {OUT_DIR}/classification_label_change.png")
    print(f"            {OUT_DIR}/speech_ratio_vs_privacy.png")
    print(f"            {OUT_DIR}/summary_table.png")
    print()

    # ── [v2 ADD] Thesis evidence ──
    print("=" * 90)
    print("STATISTICAL COMPARISON — Memory vs No-Memory (95% CI + significance)")
    print("=" * 90)
    stats_memory_vs_nomemory(paper_chunks, OUT_DIR)
    print("\n" + "=" * 90)
    print("LLM RUN PROFILE — Claude Haiku 4.5 (tokens / cost / latency)")
    print("=" * 90)
    llm_run_profile(data, labels, OUT_DIR)
    print()

    # ── [v3 ADD] Easier-to-read alternatives showing per-chunk detail ──
    print("=" * 90)
    print("ALT FIGURES — per-chunk box plots + speech-ratio grouped bars")
    print("=" * 90)
    plot_tradeoff_box(data, labels, OUT_DIR)
    plot_speechratio_bars(data, labels, OUT_DIR)
    print()

    # ── [v4 ADD] Density / facet / alpha alternatives ──
    print("=" * 90)
    print("ALT FIGURES v2 — hexbin density, alpha scatter, speech-ratio facets")
    print("=" * 90)
    plot_tradeoff_density(data, labels, OUT_DIR)
    plot_tradeoff_alpha_scatter(data, labels, OUT_DIR)
    plot_speechratio_facets(data, labels, OUT_DIR)
    print()

    total = sum(1 for f in os.listdir(OUT_DIR) if f.endswith(".png"))
    print(f"\nDone — {total} charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
