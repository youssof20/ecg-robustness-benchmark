"""
Phase 1 — Data Pipeline for ECG Robustness Benchmark.

Loads MIT-BIH Arrhythmia Database, extracts 280-sample beat segments centered on
R-peaks, maps annotations to AAMI 5-class scheme, balances by undersampling N,
splits by record (70/15/15), and saves train/val/test numpy arrays.
"""

from pathlib import Path
import numpy as np
import wfdb
from collections import Counter

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mitdb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
RECORDS_FILE = DATA_DIR / "RECORDS"
WINDOW_LEN = 280
FS = 360
AAMI_N = {"N", "L", "R", "e", "j"}
AAMI_S = {"A", "a", "J", "S"}
AAMI_V = {"V", "E"}
AAMI_F = {"F"}
AAMI_Q = {"/", "f", "Q"}
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def get_record_list() -> list[str]:
    """Read the list of MIT-BIH record names from RECORDS file."""
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"RECORDS file not found at {RECORDS_FILE}. "
            "Ensure data/mitdb/ exists and contains the MIT-BIH database."
        )
    records = [line.strip() for line in RECORDS_FILE.read_text().strip().splitlines()]
    # Exclude non-numeric entries (e.g. directory names) — MIT-BIH records are numeric
    return [r for r in records if r.isdigit()]


def annotation_to_aami(symbol: str) -> str | None:
    """
    Map a single WFDB beat annotation symbol to AAMI class (N, S, V, F, Q).
    Returns None for non-beat annotations (e.g. rhythm, noise) which should be skipped.
    """
    if not symbol or len(symbol) == 0:
        return None
    s = symbol.strip().upper() if symbol.strip() else symbol
    # WFDB can return symbols as single char; handle both
    if s in AAMI_N or symbol in AAMI_N:
        return "N"
    if s in AAMI_S or symbol in AAMI_S:
        return "S"
    if s in AAMI_V or symbol in AAMI_V:
        return "V"
    if s in AAMI_F or symbol in AAMI_F:
        return "F"
    if s in AAMI_Q or symbol in AAMI_Q:
        return "Q"
    # Non-beat (e.g. '!', '?', '[', ']', '~', '|', 'p', 't') — skip
    return None


def extract_segments_for_record(record_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one MIT-BIH record, get beat annotations, and extract 280-sample windows
    centered on each R-peak. Returns (segments, labels) for that record.
    Labels are integer indices 0..4 for N, S, V, F, Q.
    """
    record_path = DATA_DIR / record_name
    if not (record_path.with_suffix(".dat").exists() or (DATA_DIR / f"{record_name}.dat").exists()):
        raise FileNotFoundError(f"Record data not found: {record_path}.dat")
    # Read signal (first channel only) and annotations
    rec = wfdb.rdrecord(str(record_path), channels=[0])
    ann = wfdb.rdann(str(record_path), "atr")
    signal = rec.p_signal.squeeze()
    length = len(signal)
    half = WINDOW_LEN // 2
    segments_list = []
    labels_list = []
    for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        aami = annotation_to_aami(symbol)
        if aami is None:
            continue
        start = sample - half
        end = sample + half
        if start < 0 or end > length:
            continue
        segment = signal[start:end]
        if segment.shape[0] != WINDOW_LEN:
            continue
        segments_list.append(segment)
        labels_list.append(CLASS_TO_IDX[aami])
    if not segments_list:
        return np.array([]).reshape(0, WINDOW_LEN), np.array([], dtype=np.int64)
    return np.stack(segments_list, axis=0), np.array(labels_list, dtype=np.int64)


def balance_by_undersampling_n(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Undersample class N so the dataset is more balanced. Strategy: set N count
    to the size of the largest non-N class (or 2x that), then randomly sample N.
    """
    rng = np.random.default_rng(random_state)
    n_class = CLASS_TO_IDX["N"]
    n_mask = y == n_class
    other_mask = ~n_mask
    n_count = n_mask.sum()
    other_counts = Counter(y[other_mask])
    if not other_counts:
        return X, y
    target_n = max(other_counts.values())
    n_indices = np.where(n_mask)[0]
    if len(n_indices) <= target_n:
        return X, y
    keep_n = rng.choice(n_indices, size=target_n, replace=False)
    keep_other = np.where(other_mask)[0]
    keep = np.sort(np.concatenate([keep_n, keep_other]))
    return X[keep], y[keep]


def split_by_record(
    record_names: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split record names into train / val / test so that no record appears in more
    than one split (avoids data leakage). Uses fixed seed for reproducibility.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(random_state)
    names = list(record_names)
    rng.shuffle(names)
    n = len(names)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    train_records = names[:n_train]
    val_records = names[n_train : n_train + n_val]
    test_records = names[n_train + n_val :]
    return train_records, val_records, test_records


def print_class_distribution(y: np.ndarray, split_name: str) -> None:
    """Print count and percentage per AAMI class for a split."""
    counts = Counter(y)
    total = len(y)
    print(f"  {split_name}: total = {total}")
    for cname in CLASS_NAMES:
        idx = CLASS_TO_IDX[cname]
        n = counts.get(idx, 0)
        pct = 100.0 * n / total if total else 0
        print(f"    {cname}: {n} ({pct:.1f}%)")


def run_data_pipeline() -> dict:
    """
    Run the full data pipeline: load all records, extract segments, AAMI labels,
    balance, split by record, save arrays. Returns a summary dict.
    """
    print("Phase 1 — Data pipeline (MIT-BIH)")
    print("=" * 50)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    record_list = get_record_list()
    print(f"Found {len(record_list)} records in {RECORDS_FILE}")

    # Collect segments and labels per record (for split-by-record)
    record_segments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rec_name in record_list:
        try:
            X_rec, y_rec = extract_segments_for_record(rec_name)
            if len(X_rec) > 0:
                record_segments[rec_name] = (X_rec, y_rec)
        except FileNotFoundError as e:
            print(f"  Skipping {rec_name}: {e}")
        except Exception as e:
            print(f"  Error processing {rec_name}: {e}")

    if not record_segments:
        raise FileNotFoundError(
            "No records could be loaded. Check that data/mitdb/ contains "
            ".dat, .hea, and .atr files for MIT-BIH records."
        )

    # Concatenate all and compute total beats before balancing
    all_X = np.concatenate([record_segments[r][0] for r in record_segments], axis=0)
    all_y = np.concatenate([record_segments[r][1] for r in record_segments], axis=0)
    total_beats_before = len(all_y)
    print(f"\nTotal beats loaded (before balancing): {total_beats_before}")
    print_class_distribution(all_y, "All records")

    # Build record -> (X, y) for splitting
    train_records, val_records, test_records = split_by_record(list(record_segments.keys()))

    def collect_split(rec_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        for r in rec_names:
            if r in record_segments:
                X_list.append(record_segments[r][0])
                y_list.append(record_segments[r][1])
        if not X_list:
            return np.array([]).reshape(0, WINDOW_LEN), np.array([], dtype=np.int64)
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    X_train_raw, y_train_raw = collect_split(train_records)
    X_val, y_val = collect_split(val_records)
    X_test, y_test = collect_split(test_records)

    # Balance only the training set (undersample N)
    X_train, y_train = balance_by_undersampling_n(X_train_raw, y_train_raw)
    print(f"\nAfter undersampling N (train only): train size {len(y_train)} (was {len(y_train_raw)})")
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_val, "Val")
    print_class_distribution(y_test, "Test")

    # Add channel dimension for PyTorch: (N, 280) -> (N, 1, 280)
    def add_channel(a: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return a.reshape(0, 1, WINDOW_LEN)
        return a[:, np.newaxis, :]

    X_train = add_channel(X_train)
    X_val = add_channel(X_val)
    X_test = add_channel(X_test)

    # Save
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    np.save(
        OUTPUT_DIR / "record_splits.npy",
        {"train": train_records, "val": val_records, "test": test_records},
        allow_pickle=True,
    )

    print(f"\nSaved arrays to {OUTPUT_DIR}")
    print(f"  X_train.npy: {X_train.shape}")
    print(f"  y_train.npy: {y_train.shape}")
    print(f"  X_val.npy:   {X_val.shape}")
    print(f"  y_val.npy:   {y_val.shape}")
    print(f"  X_test.npy:  {X_test.shape}")
    print(f"  y_test.npy:  {y_test.shape}")

    summary = {
        "total_beats_before_balance": total_beats_before,
        "train_size_before_balance": len(y_train_raw),
        "train_size_after_balance": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "n_records": len(record_segments),
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "n_test_records": len(test_records),
        "output_dir": str(OUTPUT_DIR),
    }
    return summary


if __name__ == "__main__":
    summary = run_data_pipeline()
    print("\n" + "\033[92m" + "PHASE 1 (data pipeline) COMPLETE" + "\033[0m")
    print("Summary:", summary)
