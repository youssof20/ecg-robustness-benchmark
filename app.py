"""
ECG Robustness Benchmark — Streamlit app (Phase 4).

Multi-page app: Signal Explorer, Live Classifier, Benchmark Results.
Run from project root: streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Plotly is required for this app. Install dependencies (e.g. `pip install -r requirements.txt`)."
    ) from e

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for this app. Install dependencies (e.g. `pip install -r requirements.txt`)."
    ) from e

try:
    import wfdb
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "WFDB is required for this app. Install dependencies (e.g. `pip install -r requirements.txt`)."
    ) from e

# Local imports (project code)
from src.models import get_model, NUM_CLASSES

st.set_page_config(
    page_title="ECG Robustness Benchmark",
    page_icon="ECG",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
NSTDB_DIR = PROJECT_ROOT / "data" / "nstdb"

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
NOISE_TYPES = ["bw", "ma", "em"]
NOISE_LABELS = {"bw": "BW (Baseline wander)", "ma": "MA (Muscle artifact)", "em": "EM (Electrode motion)"}
SNR_LEVELS = [24, 18, 12, 6, 0, -6]
CLEAN_SNR = 999
WINDOW_LEN = 280
EPS = 1e-8


@dataclass(frozen=True)
class BeatSelection:
    beat_idx: int
    noise_type: str
    snr_db: int


def _require_file(path: Path, hint: str) -> bool:
    if path.exists():
        return True
    st.error(f"Missing required file: `{path}`\n\n{hint}")
    return False


@st.cache_resource
def load_test_arrays() -> tuple[np.ndarray, np.ndarray]:
    x_path = RESULTS_DIR / "X_test.npy"
    y_path = RESULTS_DIR / "y_test.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing `{x_path}` or `{y_path}`. Run Phase 1 first (and keep outputs/ in place)."
        )
    X_test = np.load(x_path).astype(np.float32)
    y_test = np.load(y_path).astype(np.int64)
    return X_test, y_test


@st.cache_resource
def load_noise_signals() -> dict[str, np.ndarray]:
    """Load full raw NSTDB noise signals (channel 0) for bw/ma/em."""
    out: dict[str, np.ndarray] = {}
    for nt in NOISE_TYPES:
        rec = wfdb.rdrecord(str(NSTDB_DIR / nt), channels=[0])
        out[nt] = rec.p_signal.squeeze().astype(np.float64)
        if out[nt].ndim != 1 or len(out[nt]) < WINDOW_LEN + 1:
            raise ValueError(f"Noise signal `{nt}` looks invalid: shape={out[nt].shape}")
    return out


@st.cache_resource
def load_models() -> dict[str, torch.nn.Module]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: dict[str, torch.nn.Module] = {}
    for name in ["SimpleCNN", "ResNet1D", "LightweightNet"]:
        ckpt_path = MODELS_DIR / f"{name}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint `{ckpt_path}`. Run Phase 2 training first.")
        model = get_model(name).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[name] = model
    return models


def ensure_session_defaults(n_beats: int) -> None:
    if "beat_idx" not in st.session_state:
        st.session_state.beat_idx = int(np.random.default_rng().integers(0, n_beats))
    if "noise_type" not in st.session_state:
        st.session_state.noise_type = "em"
    if "snr_db" not in st.session_state:
        st.session_state.snr_db = 24
    if "noise_start" not in st.session_state:
        # Keep the same noise window while the user changes SNR (so changes are interpretable).
        st.session_state.noise_start = {nt: None for nt in NOISE_TYPES}


def get_clean_segment(X_test: np.ndarray, beat_idx: int) -> np.ndarray:
    seg = X_test[beat_idx]
    if seg.ndim == 2:
        seg = seg[0]
    return seg.astype(np.float64)


def choose_noise_start(noise_signal: np.ndarray, beat_idx: int) -> int:
    max_start = len(noise_signal) - WINDOW_LEN
    # Deterministic per beat, so reloading the page keeps the same noise slice.
    rng = np.random.default_rng(10_000 + int(beat_idx))
    return int(rng.integers(0, max_start + 1))


def mix_at_snr(clean_segment: np.ndarray, noise_signal: np.ndarray, snr_db: int, start: int) -> np.ndarray:
    """
    Same formula as src/noise_mixer.py:
      noisy = clean + (std_clean / (std_noise + eps)) * 10^(-SNR_dB/20) * noise_window
    """
    clean_segment = np.asarray(clean_segment, dtype=np.float64).reshape(-1)
    noise_window = np.asarray(noise_signal[start : start + WINDOW_LEN], dtype=np.float64).reshape(-1)
    if len(clean_segment) != WINDOW_LEN or len(noise_window) != WINDOW_LEN:
        raise ValueError("Expected 280-sample clean segment and noise window.")

    std_clean = float(clean_segment.std())
    std_noise = float(noise_window.std())
    std_clean = std_clean if std_clean >= EPS else EPS
    std_noise = std_noise if std_noise >= EPS else EPS
    scale = std_clean / std_noise
    snr_factor = 10.0 ** (-float(snr_db) / 20.0)
    noisy = clean_segment + noise_window * scale * snr_factor
    return noisy.astype(np.float32)


def predict_with_confidence(
    model: torch.nn.Module, x_1x1x280: np.ndarray
) -> tuple[int, float, np.ndarray]:
    device = next(model.parameters()).device
    x = torch.from_numpy(x_1x1x280.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred, conf, probs


def format_label(y: int) -> str:
    return CLASS_NAMES[int(y)] if 0 <= int(y) < len(CLASS_NAMES) else str(int(y))


def plot_signal_overlay(clean: np.ndarray, noisy: np.ndarray, title: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=clean, mode="lines", name="Clean", line=dict(width=3, color="#4C78A8")))
    fig.add_trace(go.Scatter(y=noisy, mode="lines", name="Noisy", line=dict(width=2, color="#F58518")))
    fig.update_layout(
        title=title,
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        height=350,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, width="stretch")


def page_signal_explorer(sel: BeatSelection) -> None:
    st.subheader("Signal Explorer")
    X_test, y_test = load_test_arrays()
    noise_signals = load_noise_signals()

    ensure_session_defaults(X_test.shape[0])

    # Controls
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        noise_type = st.selectbox(
            "Noise type",
            options=NOISE_TYPES,
            format_func=lambda x: NOISE_LABELS.get(x, x),
            index=NOISE_TYPES.index(sel.noise_type),
            key="noise_type",
        )
    with c2:
        snr_db = st.selectbox("SNR (dB)", options=SNR_LEVELS, index=SNR_LEVELS.index(sel.snr_db), key="snr_db")
    with c3:
        if st.button("Resample beat", width="stretch"):
            st.session_state.beat_idx = int(np.random.default_rng().integers(0, X_test.shape[0]))
            st.session_state.noise_start = {nt: None for nt in NOISE_TYPES}

    beat_idx = int(st.session_state.beat_idx)
    y_true = int(y_test[beat_idx])
    clean = get_clean_segment(X_test, beat_idx)

    # Stable noise window per beat + noise type
    if st.session_state.noise_start.get(noise_type) is None:
        st.session_state.noise_start[noise_type] = choose_noise_start(noise_signals[noise_type], beat_idx)
    start = int(st.session_state.noise_start[noise_type])
    noisy = mix_at_snr(clean, noise_signals[noise_type], int(snr_db), start)

    title = f"Beat #{beat_idx} — True label: {format_label(y_true)} — {NOISE_LABELS.get(noise_type, noise_type)}, {snr_db} dB"
    plot_signal_overlay(clean.astype(np.float32), noisy, title=title)

    with st.expander("Beat stats", expanded=False):
        st.write(
            pd.DataFrame(
                {
                    "signal": ["clean", "noisy"],
                    "mean": [float(clean.mean()), float(noisy.mean())],
                    "std": [float(clean.std()), float(noisy.std())],
                }
            )
        )


def page_live_classifier(sel: BeatSelection) -> None:
    st.subheader("Live Classifier")
    X_test, y_test = load_test_arrays()
    noise_signals = load_noise_signals()
    models = load_models()

    ensure_session_defaults(X_test.shape[0])

    # SNR slider updates session state (shared with Signal Explorer)
    st.slider(
        "SNR (dB)",
        min_value=min(SNR_LEVELS),
        max_value=max(SNR_LEVELS),
        value=int(st.session_state.snr_db),
        step=6,
        key="snr_db",
        help="Drag to see predictions update as noise increases/decreases.",
    )

    beat_idx = int(st.session_state.beat_idx)
    noise_type = str(st.session_state.noise_type)
    snr_db = int(st.session_state.snr_db)
    y_true = int(y_test[beat_idx])

    clean = get_clean_segment(X_test, beat_idx)
    if st.session_state.noise_start.get(noise_type) is None:
        st.session_state.noise_start[noise_type] = choose_noise_start(noise_signals[noise_type], beat_idx)
    start = int(st.session_state.noise_start[noise_type])
    noisy = mix_at_snr(clean, noise_signals[noise_type], snr_db, start)

    # Quick signal view
    plot_signal_overlay(
        clean.astype(np.float32),
        noisy,
        title=f"Beat #{beat_idx} — True label: {format_label(y_true)} — {NOISE_LABELS.get(noise_type, noise_type)}, {snr_db} dB",
    )

    x_clean = clean.astype(np.float32)[None, None, :]
    x_noisy = noisy.astype(np.float32)[None, None, :]

    rows = []
    prob_rows = []
    for name, model in models.items():
        pred_c, conf_c, probs_c = predict_with_confidence(model, x_clean)
        pred_n, conf_n, probs_n = predict_with_confidence(model, x_noisy)
        rows.append(
            {
                "model": name,
                "clean_pred": format_label(pred_c),
                "clean_conf": conf_c,
                "noisy_pred": format_label(pred_n),
                "noisy_conf": conf_n,
            }
        )
        for cls_i, cls_name in enumerate(CLASS_NAMES):
            prob_rows.append({"model": name, "condition": "clean", "class": cls_name, "prob": float(probs_c[cls_i])})
            prob_rows.append({"model": name, "condition": "noisy", "class": cls_name, "prob": float(probs_n[cls_i])})

    df = pd.DataFrame(rows)

    def _style_row(r: pd.Series) -> list[str]:
        ok_clean = r["clean_pred"] == format_label(y_true)
        ok_noisy = r["noisy_pred"] == format_label(y_true)
        return [
            "",
            "background-color: #163a24" if ok_clean else "background-color: #4a1b1b",
            "",
            "background-color: #163a24" if ok_noisy else "background-color: #4a1b1b",
            "",
        ]

    st.markdown(f"**True label:** `{format_label(y_true)}`")
    st.dataframe(
        df.style.apply(_style_row, axis=1).format({"clean_conf": "{:.3f}", "noisy_conf": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )

    probs_long = pd.DataFrame(prob_rows)
    fig = px.bar(
        probs_long,
        x="class",
        y="prob",
        color="condition",
        barmode="group",
        facet_col="model",
        category_orders={"class": CLASS_NAMES, "model": list(models.keys()), "condition": ["clean", "noisy"]},
        title="Softmax probabilities (clean vs noisy)",
        template="plotly_dark",
        height=380,
    )
    fig.update_yaxes(range=[0, 1])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, width="stretch")


def compute_robustness_auc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustness score per noise type:
      AUC over SNR levels (-6..24) for macro_f1, normalized by span (30 dB).
    """
    norm_span = 24 - (-6)
    out_rows = []
    for noise_type in NOISE_TYPES:
        sub = df[(df["noise_type"] == noise_type) & (df["snr_db"] != CLEAN_SNR)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("snr_db")
        for model in sorted(sub["model"].unique()):
            m = sub[sub["model"] == model]
            auc = float(np.trapezoid(m["macro_f1"].values, m["snr_db"].values) / norm_span)
            out_rows.append({"model": model, "noise_type": noise_type, "robustness_auc": auc})
    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    return out.pivot(index="model", columns="noise_type", values="robustness_auc").reset_index()


def page_benchmark_results() -> None:
    st.subheader("Benchmark Results")

    csv_path = RESULTS_DIR / "benchmark_results.csv"
    if not _require_file(csv_path, "Run `python src/benchmark.py` first to generate this CSV."):
        return
    df = pd.read_csv(csv_path)
    if "noise_type" not in df.columns:
        st.error("`benchmark_results.csv` is missing the `noise_type` column. Re-run the benchmark.")
        return

    # Static figures
    st.markdown("**Pre-computed figures**")
    fig_files = [
        ("Degradation curves", FIGURES_DIR / "degradation_curves.png"),
        ("Robustness heatmap", FIGURES_DIR / "robustness_heatmap.png"),
        ("Accuracy drop", FIGURES_DIR / "accuracy_drop.png"),
        ("Robustness score", FIGURES_DIR / "robustness_score.png"),
    ]
    for label, path in fig_files:
        if path.exists():
            st.image(str(path), caption=label, width="stretch")
        else:
            st.warning(f"Missing figure `{path}` (run `python src/visualize.py`).")

    st.divider()

    # Interactive degradation curve
    st.markdown("**Interactive degradation curve**")
    noise_type = st.selectbox(
        "Filter noise type",
        options=NOISE_TYPES,
        format_func=lambda x: NOISE_LABELS.get(x, x),
        index=NOISE_TYPES.index("em"),
    )
    clean = df[(df["noise_type"] == "clean") & (df["snr_db"] == CLEAN_SNR)].copy()
    sub = df[df["noise_type"] == noise_type].copy()
    if sub.empty or clean.empty:
        st.warning("Not enough rows to plot this noise type. Re-run benchmark.")
        return

    plot_rows = []
    for model in sorted(df["model"].unique()):
        clean_row = clean[clean["model"] == model].iloc[0].to_dict()
        clean_row["snr_label"] = "clean"
        clean_row["snr_plot"] = CLEAN_SNR
        plot_rows.append(clean_row)
        for snr in SNR_LEVELS:
            r = sub[(sub["model"] == model) & (sub["snr_db"] == snr)].iloc[0].to_dict()
            r["snr_label"] = str(int(snr))
            r["snr_plot"] = int(snr)
            plot_rows.append(r)

    plot_df = pd.DataFrame(plot_rows)
    # Plotly wants numeric x; we keep CLEAN_SNR=999 and label it "clean"
    fig = px.line(
        plot_df.sort_values(["model", "snr_plot"], ascending=[True, False]),
        x="snr_plot",
        y="macro_f1",
        color="model",
        markers=True,
        title=f"Macro F1 vs SNR — {NOISE_LABELS.get(noise_type, noise_type)}",
        template="plotly_dark",
        height=380,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[CLEAN_SNR] + SNR_LEVELS,
        ticktext=["clean"] + [str(s) for s in SNR_LEVELS],
        title="SNR (dB)",
    )
    fig.update_yaxes(range=[0, 1], title="Macro F1")
    st.plotly_chart(fig, width="stretch")

    # Robustness score table
    st.markdown("**Robustness score (AUC) table**")
    auc_table = compute_robustness_auc(df)
    if auc_table.empty:
        st.warning("Could not compute robustness score table.")
    else:
        st.dataframe(auc_table.style.format({t: "{:.3f}" for t in NOISE_TYPES}), width="stretch", hide_index=True)

    st.markdown("**Key Findings**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**LightweightNet** is the best overall performer with only **8,429 parameters**.")
    with c2:
        st.info("**EM noise** degrades performance most gracefully across models.")
    with c3:
        st.info("**S and F classes** remain near-zero F1 across noise levels (hardest beats).")


def main() -> None:
    st.title("ECG Robustness Benchmark")
    st.markdown(
        "Benchmarking arrhythmia classifier robustness under real-world noise (BW/MA/EM) "
        "at controlled SNR levels. This app only uses **pre-computed artifacts** — it does not train models."
    )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Page",
        ["Signal Explorer", "Live Classifier", "Benchmark Results"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project**: ECG Robustness Benchmark")
    st.sidebar.caption("Paper + code release companion app.")
    st.sidebar.markdown("**GitHub**: `https://github.com/<your-org>/<your-repo>`")

    # Lightweight validation hints (no heavy loading here).
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Artifacts**")
    st.sidebar.write(f"- `X_test.npy`: {'OK' if (RESULTS_DIR / 'X_test.npy').exists() else 'missing'}")
    st.sidebar.write(f"- `y_test.npy`: {'OK' if (RESULTS_DIR / 'y_test.npy').exists() else 'missing'}")
    st.sidebar.write(f"- `benchmark_results.csv`: {'OK' if (RESULTS_DIR / 'benchmark_results.csv').exists() else 'missing'}")

    # Shared selection state
    if (RESULTS_DIR / "X_test.npy").exists():
        X_test, _ = load_test_arrays()
        ensure_session_defaults(X_test.shape[0])
    sel = BeatSelection(
        beat_idx=int(st.session_state.get("beat_idx", 0)),
        noise_type=str(st.session_state.get("noise_type", "em")),
        snr_db=int(st.session_state.get("snr_db", 24)),
    )

    if page == "Signal Explorer":
        if not _require_file(RESULTS_DIR / "X_test.npy", "Run Phase 1 to generate test arrays."):
            return
        if not _require_file(NSTDB_DIR / "em.dat", "Ensure NSTDB files exist in `data/nstdb/`."):
            return
        page_signal_explorer(sel)
    elif page == "Live Classifier":
        if not _require_file(MODELS_DIR / "SimpleCNN_best.pt", "Run Phase 2 to train and save checkpoints."):
            return
        if not _require_file(RESULTS_DIR / "X_test.npy", "Run Phase 1 to generate test arrays."):
            return
        page_live_classifier(sel)
    else:
        page_benchmark_results()


if __name__ == "__main__":
    main()
