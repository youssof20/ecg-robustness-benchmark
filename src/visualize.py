"""
Phase 3 — Visualize robustness benchmark results.

Reads benchmark_results.csv (with noise_type: clean, bw, ma, em) and generates
four figures: degradation curves (3 subplots), heatmap (3), accuracy drop (3), robustness score.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SNR_LEVELS = [24, 18, 12, 6, 0, -6]
CLEAN_SNR = 999
MODEL_NAMES = ["SimpleCNN", "ResNet1D", "LightweightNet"]
NOISE_TYPES = ["bw", "ma", "em"]
NOISE_LABELS = {"bw": "Baseline wander", "ma": "Muscle artifact", "em": "Electrode motion"}


def load_benchmark_df():
    path = RESULTS_DIR / "benchmark_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run benchmark.py first. Missing: {path}")
    df = pd.read_csv(path)
    if "noise_type" not in df.columns:
        df["noise_type"] = "em"
    return df


def plot_degradation_curves(df: pd.DataFrame, out_path: Path) -> None:
    """Subplots per noise type (bw, ma, em); each: x=SNR, y=Macro F1, 3 lines per model."""
    x_order = [CLEAN_SNR] + SNR_LEVELS
    x_labels = ["clean", "24", "18", "12", "6", "0", "-6"]
    x_pos = np.arange(len(x_order))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    clean = df[(df["noise_type"] == "clean") & (df["snr_db"] == CLEAN_SNR)].copy()
    types_present = [t for t in NOISE_TYPES if t in df["noise_type"].values]
    n_axes = max(1, len(types_present))
    fig, axes = plt.subplots(1, n_axes, figsize=(4 * n_axes, 4), sharey=True)
    if n_axes == 1:
        axes = [axes]
    for ax, noise_type in zip(axes, types_present):
        sub = df[df["noise_type"] == noise_type].copy()
        for i, model in enumerate(MODEL_NAMES):
            noisy_m = sub[sub["model"] == model].set_index("snr_db")
            m = noisy_m.reindex(x_order)
            if (clean["model"] == model).any():
                m.loc[CLEAN_SNR, "macro_f1"] = float(clean[clean["model"] == model]["macro_f1"].values[0])
            ax.plot(x_pos, m["macro_f1"].values, "o-", label=model, color=colors[i], linewidth=2, markersize=6)
        ax.axhline(0.2, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Macro F1" if ax == axes[0] else "")
        ax.set_ylim(0, 1.02)
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type))
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("ECG Classifier Robustness by Noise Type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_robustness_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmaps (one per noise type): rows=models, cols=clean+6 SNR."""
    x_order = [CLEAN_SNR] + SNR_LEVELS
    x_labels = ["clean", "24", "18", "12", "6", "0", "-6"]
    clean = df[(df["noise_type"] == "clean") & (df["snr_db"] == CLEAN_SNR)].copy()
    types_present = [t for t in NOISE_TYPES if t in df["noise_type"].values]
    n_axes = max(1, len(types_present))
    fig, axes = plt.subplots(1, n_axes, figsize=(4 * n_axes, 3), sharey=True)
    if n_axes == 1:
        axes = [axes]
    for ax, noise_type in zip(axes, types_present):
        sub = df[df["noise_type"] == noise_type]
        data = np.zeros((len(MODEL_NAMES), len(x_order)))
        for i, model in enumerate(MODEL_NAMES):
            m = sub[sub["model"] == model].set_index("snr_db")
            for j, snr in enumerate(x_order):
                if snr == CLEAN_SNR:
                    data[i, j] = float(clean[clean["model"] == model]["macro_f1"].values[0])
                else:
                    data[i, j] = float(m.loc[snr, "macro_f1"])
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(MODEL_NAMES)))
        ax.set_yticklabels(MODEL_NAMES)
        for i in range(len(MODEL_NAMES)):
            for j in range(len(x_order)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type))
    if n_axes == 1:
        plt.colorbar(im, ax=axes[0], label="Macro F1")
    else:
        plt.colorbar(im, ax=axes, label="Macro F1", shrink=0.6)
    plt.suptitle("Robustness Heatmap — Macro F1 by Model and Noise Level")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_accuracy_drop(df: pd.DataFrame, out_path: Path) -> None:
    """Subplots per noise type: grouped bars at clean, 24, 0, -6 dB."""
    key_snrs = [CLEAN_SNR, 24, 0, -6]
    key_labels = ["clean", "24 dB", "0 dB", "-6 dB"]
    clean = df[(df["noise_type"] == "clean") & (df["snr_db"] == CLEAN_SNR)].copy()
    types_present = [t for t in NOISE_TYPES if t in df["noise_type"].values]
    n_axes = max(1, len(types_present))
    fig, axes = plt.subplots(1, n_axes, figsize=(4 * n_axes, 4), sharey=True)
    if n_axes == 1:
        axes = [axes]
    width = 0.2
    x = np.arange(len(MODEL_NAMES))
    for ax, noise_type in zip(axes, types_present):
        sub = df[df["noise_type"] == noise_type]
        for k, (snr, label) in enumerate(zip(key_snrs, key_labels)):
            accs = []
            for model in MODEL_NAMES:
                if snr == CLEAN_SNR:
                    accs.append(float(clean[clean["model"] == model]["accuracy"].values[0]))
                else:
                    accs.append(float(sub[(sub["model"] == model) & (sub["snr_db"] == snr)]["accuracy"].values[0]))
            offset = (k - 1.5) * width
            ax.bar(x + offset, accs, width, label=label)
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_NAMES)
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type))
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.02)
    plt.suptitle("Accuracy at Key SNR Levels")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_robustness_score(df: pd.DataFrame, out_path: Path) -> None:
    """Robustness score = AUC of macro F1 over SNR (-6 to 24). One bar per model (average over noise types)."""
    norm_span = 24 - (-6)
    scores = {m: [] for m in MODEL_NAMES}
    types_present = [t for t in NOISE_TYPES if t in df["noise_type"].values]
    if not types_present:
        types_present = ["em"]
    for noise_type in types_present:
        sub = df[(df["noise_type"] == noise_type) & (df["snr_db"] != CLEAN_SNR)].sort_values("snr_db", ascending=True)
        for model in MODEL_NAMES:
            m = sub[sub["model"] == model]
            auc = np.trapezoid(m["macro_f1"].values, m["snr_db"].values)
            scores[model].append(auc / norm_span)
    avg_scores = {m: np.mean(scores[m]) for m in MODEL_NAMES}
    ranked = sorted(avg_scores.items(), key=lambda x: -x[1])
    names = [x[0] for x in ranked]
    vals = [x[1] for x in ranked]
    colors = ["#2ca02c"] + ["#7f7f7f"] * (len(names) - 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Robustness score (AUC normalized, avg over bw/ma/em)")
    ax.set_title("Overall Robustness Score (Area Under Degradation Curve)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def run_visualize() -> None:
    print("Phase 3 — Visualize")
    print("=" * 50)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_benchmark_df()
    print(f"Loaded {len(df)} rows from benchmark_results.csv")
    plot_degradation_curves(df, FIGURES_DIR / "degradation_curves.png")
    plot_robustness_heatmap(df, FIGURES_DIR / "robustness_heatmap.png")
    plot_accuracy_drop(df, FIGURES_DIR / "accuracy_drop.png")
    plot_robustness_score(df, FIGURES_DIR / "robustness_score.png")
    print("\n\033[92mPHASE 3 (visualize) COMPLETE\033[0m")


if __name__ == "__main__":
    run_visualize()
