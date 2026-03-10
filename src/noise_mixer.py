"""
Generate noisy test sets by mixing MIT-BIH clean segments with NSTDB raw noise
at controlled SNR levels. Ensures noisy signals stay on the same scale as training data.

Formula: noisy = clean + (std_clean / (std_noise + eps)) * 10^(-SNR_dB/20) * noise_window
"""

from pathlib import Path
import numpy as np
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
NSTDB_DIR = PROJECT_ROOT / "data" / "nstdb"

NOISE_TYPES = ["bw", "ma", "em"]
SNR_LEVELS = [24, 18, 12, 6, 0, -6]
WINDOW_LEN = 280
EPS = 1e-8
RNG = np.random.default_rng(42)


def load_noise_signal(noise_type: str) -> np.ndarray:
    """Load full raw noise recording (first channel) from data/nstdb/."""
    path = NSTDB_DIR / noise_type
    rec = wfdb.rdrecord(str(path), channels=[0])
    return rec.p_signal.squeeze().astype(np.float64)


def mix_at_snr(
    clean_segment: np.ndarray,
    noise_signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mix one 280-sample clean segment with a random 280-sample window from noise
    at target SNR. Returns noisy segment on same scale as clean.
    """
    std_clean = clean_segment.std()
    if std_clean < EPS:
        std_clean = EPS
    max_start = len(noise_signal) - WINDOW_LEN
    if max_start <= 0:
        raise ValueError(f"Noise signal too short: {len(noise_signal)}")
    start = rng.integers(0, max_start + 1)
    noise_window = noise_signal[start : start + WINDOW_LEN].copy()
    std_noise = noise_window.std()
    if std_noise < EPS:
        std_noise = EPS
    scale = std_clean / std_noise
    snr_factor = 10.0 ** (-snr_db / 20.0)
    noise_scaled = noise_window * scale * snr_factor
    return (clean_segment + noise_scaled).astype(np.float32)


def run_noise_mixer() -> list[tuple[str, int, Path]]:
    """
    Load X_test, y_test; for each noise type and SNR, generate noisy test set
    by mixing with NSTDB raw noise. Save .npy and return list of (noise_type, snr, path).
    """
    print("Noise mixer — generate noisy test sets (same scale as MIT-BIH)")
    print("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_test = np.load(RESULTS_DIR / "X_test.npy").astype(np.float64)
    y_test = np.load(RESULTS_DIR / "y_test.npy")
    n_beats = X_test.shape[0]
    if X_test.ndim == 3:
        clean_segments = X_test[:, 0, :]
    else:
        clean_segments = X_test
    print(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Load all three noise signals
    noise_signals = {}
    for nt in NOISE_TYPES:
        noise_signals[nt] = load_noise_signal(nt)
        print(f"  Loaded {nt}: {len(noise_signals[nt])} samples")

    # Reference clean stats (from user: mean=-0.2046, std=0.4587)
    clean_mean_ref = float(np.mean(clean_segments))
    clean_std_ref = float(np.std(clean_segments))
    print(f"\nClean test (reference): mean={clean_mean_ref:.4f}, std={clean_std_ref:.4f}")

    created = []
    diagnostic_rows = []

    for noise_type in NOISE_TYPES:
        noise_sig = noise_signals[noise_type]
        for snr_db in SNR_LEVELS:
            noisy = np.empty((n_beats, 1, WINDOW_LEN), dtype=np.float32)
            for i in range(n_beats):
                seg = clean_segments[i]
                noisy[i, 0, :] = mix_at_snr(seg, noise_sig, snr_db, RNG)
            out_name = f"noisy_test_{noise_type}_{snr_db}dB.npy"
            out_path = RESULTS_DIR / out_name
            np.save(out_path, noisy)
            created.append((noise_type, snr_db, out_path))
            # Diagnostic: first 100 segments
            first100 = noisy[:100].reshape(-1)
            mean_100 = float(np.mean(first100))
            std_100 = float(np.std(first100))
            diagnostic_rows.append((noise_type, snr_db, mean_100, std_100))

    # Diagnostic table
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Mean and std of first 100 noisy segments (should be close to clean)")
    print("=" * 60)
    print(f"Clean (reference): mean={clean_mean_ref:.4f}, std={clean_std_ref:.4f}")
    print(f"{'Noise':<6} {'SNR':>6} {'mean':>10} {'std':>10}")
    print("-" * 36)
    for noise_type, snr_db, mean_100, std_100 in diagnostic_rows:
        print(f"{noise_type:<6} {snr_db:>5} dB {mean_100:>10.4f} {std_100:>10.4f}")
    print("=" * 60)

    # Figure: 3 rows (noise types), 7 columns (clean + 6 SNR)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 7, figsize=(14, 6), sharex=True, sharey=False)
    beat_idx = 0
    for row, noise_type in enumerate(NOISE_TYPES):
        # Column 0: clean
        axes[row, 0].plot(clean_segments[beat_idx], color="C0")
        axes[row, 0].set_ylabel(noise_type.upper())
        axes[row, 0].set_title("clean")
        axes[row, 0].grid(True, alpha=0.3)
        for col, snr_db in enumerate(SNR_LEVELS):
            noisy = np.load(RESULTS_DIR / f"noisy_test_{noise_type}_{snr_db}dB.npy")
            axes[row, col + 1].plot(noisy[beat_idx, 0, :], color="C1")
            axes[row, col + 1].set_title(f"{snr_db} dB")
            axes[row, col + 1].grid(True, alpha=0.3)
    axes[0, 0].set_title("clean")
    axes[2, 0].set_xlabel("Sample")
    axes[2, 3].set_xlabel("Sample")
    plt.suptitle("Noise examples: one beat — clean then increasing noise (left to right)")
    plt.tight_layout()
    out_fig = FIGURES_DIR / "noise_examples.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\nSaved {out_fig}")

    # List output files
    print("\nOutput files created:")
    print("-" * 50)
    for noise_type, snr_db, path in created:
        arr = np.load(path)
        print(f"  {path.name}: shape {arr.shape}")
    print("-" * 50)
    print("\033[92mNoise mixer complete.\033[0m")
    return created


if __name__ == "__main__":
    run_noise_mixer()
