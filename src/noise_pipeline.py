"""
Phase 1 — Noise Pipeline for ECG Robustness Benchmark.

Loads pre-mixed noisy records from MIT-BIH Noise Stress Test Database (NSTDB),
extracts 280-sample beat segments at the same R-peak positions as the clean
MIT-BIH records 118 and 119. Saves one .npy file per (record, noise_type, snr).
"""

from pathlib import Path
import re
import sys
import numpy as np
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from data_pipeline import (
    DATA_DIR as MITDB_DIR,
    OUTPUT_DIR,
    WINDOW_LEN,
    annotation_to_aami,
    CLASS_TO_IDX,
)

# NSTDB paths
NSTDB_DIR = PROJECT_ROOT / "data" / "nstdb"
NSTDB_RECORDS_FILE = NSTDB_DIR / "RECORDS"

# NSTDB naming: [record][noise_type][snr] e.g. 118e00, 118e_6
# noise_type: e=electrode motion (em), m=muscle artifact (ma), b=baseline wander (bw)
# SNR: 00=0dB, 06=6dB, 12=12dB, 18=18dB, 24=24dB, _6=-6dB
NOISE_TYPE_CODE_TO_NAME = {"e": "em", "m": "ma", "b": "bw"}
SNR_SUFFIX_TO_DB = {"24": 24, "18": 18, "12": 12, "06": 6, "00": 0, "_6": -6}


def parse_noisy_record_name(record_name: str) -> tuple[str, str, int] | None:
    """
    Parse NSTDB record name into (base_record, noise_type_name, snr_db).
    e.g. '118e24' -> ('118', 'em', 24), '119e_6' -> ('119', 'em', -6).
    Returns None if the name does not match the pre-mixed pattern.
    """
    # Pattern: 1-3 digit record + single letter noise type + SNR suffix
    match = re.match(r"^(\d{2,3})([emb])(24|18|12|06|00|_6)$", record_name)
    if not match:
        return None
    base, code, snr_suffix = match.groups()
    noise_name = NOISE_TYPE_CODE_TO_NAME.get(code)
    if noise_name is None:
        return None
    snr_db = SNR_SUFFIX_TO_DB.get(snr_suffix)
    if snr_db is None:
        return None
    return (base, noise_name, snr_db)


def get_noisy_record_list() -> list[tuple[str, str, str, int]]:
    """
    Read NSTDB RECORDS and return list of (record_name, base_record, noise_type, snr_db)
    for pre-mixed noisy records only (exclude raw noise records like 'bw', 'em', 'ma').
    """
    if not NSTDB_RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"NSTDB RECORDS not found at {NSTDB_RECORDS_FILE}. "
            "Ensure data/nstdb/ exists and contains the NSTDB database."
        )
    lines = [line.strip() for line in NSTDB_RECORDS_FILE.read_text().strip().splitlines()]
    result = []
    for name in lines:
        parsed = parse_noisy_record_name(name)
        if parsed is not None:
            base_record, noise_type, snr_db = parsed
            result.append((name, base_record, noise_type, snr_db))
    return result


def get_beat_positions_from_clean_record(base_record: str) -> np.ndarray:
    """
    Load annotations for a clean MIT-BIH record (118 or 119) and return array of
    sample indices for beats that have valid AAMI labels and enough margin for
    a 280-sample window centered on the R-peak.
    """
    record_path = MITDB_DIR / base_record
    ann = wfdb.rdann(str(record_path), "atr")
    half = WINDOW_LEN // 2
    # Get signal length from header to check bounds
    rec = wfdb.rdrecord(str(record_path), channels=[0])
    length = rec.p_signal.shape[0]
    positions = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if annotation_to_aami(symbol) is None:
            continue
        if sample - half < 0 or sample + half > length:
            continue
        positions.append(sample)
    return np.array(positions, dtype=np.int64)


def get_beat_labels_from_clean_record(base_record: str) -> np.ndarray:
    """
    Load annotations for a clean MIT-BIH record and return AAMI class indices
    (0=N, 1=S, 2=V, 3=F, 4=Q) for each beat, in the same order as
    get_beat_positions_from_clean_record and the saved noisy segments.
    """
    record_path = MITDB_DIR / base_record
    ann = wfdb.rdann(str(record_path), "atr")
    half = WINDOW_LEN // 2
    rec = wfdb.rdrecord(str(record_path), channels=[0])
    length = rec.p_signal.shape[0]
    labels = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        aami = annotation_to_aami(symbol)
        if aami is None:
            continue
        if sample - half < 0 or sample + half > length:
            continue
        labels.append(CLASS_TO_IDX[aami])
    return np.array(labels, dtype=np.int64)


def extract_noisy_segments(
    noisy_record_name: str, beat_positions: np.ndarray
) -> np.ndarray:
    """
    Load a noisy NSTDB record and extract 280-sample segments centered at each
    beat position. Returns array of shape (n_beats, 280).
    """
    record_path = NSTDB_DIR / noisy_record_name
    rec = wfdb.rdrecord(str(record_path), channels=[0])
    signal = rec.p_signal.squeeze()
    length = len(signal)
    half = WINDOW_LEN // 2
    segments = []
    for pos in beat_positions:
        start = pos - half
        end = pos + half
        if start < 0 or end > length:
            continue
        seg = signal[start:end]
        if seg.shape[0] == WINDOW_LEN:
            segments.append(seg)
    if not segments:
        return np.array([]).reshape(0, WINDOW_LEN)
    return np.stack(segments, axis=0)


def run_noise_pipeline() -> list[tuple[str, str, int, Path]]:
    """
    For each pre-mixed noisy record in NSTDB, load clean beat positions from
    the corresponding MIT-BIH record, extract segments from the noisy signal,
    and save to outputs/results/noisy_segments_{record}_{noise_type}_{snr}.npy.
    Returns list of (record, noise_type, snr_db, output_path) for summary.
    """
    print("Phase 1 — Noise pipeline (NSTDB)")
    print("=" * 50)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    noisy_records = get_noisy_record_list()
    if not noisy_records:
        print(
            "No pre-mixed noisy records found in NSTDB RECORDS. "
            "Expected names like 118e24, 119e00, 118e_6."
        )
        return []

    # Cache beat positions and labels for base records 118 and 119
    beat_positions_cache: dict[str, np.ndarray] = {}
    beat_labels_cache: dict[str, np.ndarray] = {}
    for rec_name, base_record, noise_type, snr_db in noisy_records:
        if base_record not in beat_positions_cache:
            try:
                beat_positions_cache[base_record] = get_beat_positions_from_clean_record(
                    base_record
                )
                beat_labels_cache[base_record] = get_beat_labels_from_clean_record(base_record)
                n = len(beat_positions_cache[base_record])
                assert n == len(beat_labels_cache[base_record]), "positions/labels length mismatch"
                print(f"  Loaded {n} beat positions + labels from MIT-BIH {base_record}")
            except FileNotFoundError as e:
                print(f"  Missing clean record {base_record}: {e}")
                continue
            except Exception as e:
                print(f"  Error loading {base_record}: {e}")
                continue

    created = []
    for rec_name, base_record, noise_type, snr_db in noisy_records:
        if base_record not in beat_positions_cache:
            continue
        noisy_path = NSTDB_DIR / rec_name
        if not (noisy_path.with_suffix(".dat").exists() or (NSTDB_DIR / f"{rec_name}.dat").exists()):
            print(f"  Missing noisy record: {noisy_path}.dat")
            continue
        try:
            positions = beat_positions_cache[base_record]
            labels = beat_labels_cache[base_record]
            segments = extract_noisy_segments(rec_name, positions)
            out_name = f"noisy_segments_{base_record}_{noise_type}_{snr_db}.npy"
            out_path = OUTPUT_DIR / out_name
            np.save(out_path, segments)
            labels_name = f"noisy_labels_{base_record}_{noise_type}_{snr_db}.npy"
            labels_path = OUTPUT_DIR / labels_name
            np.save(labels_path, labels)
            created.append((base_record, noise_type, snr_db, out_path))
            print(f"  Saved {out_name}: shape {segments.shape}  |  {labels_name}: shape {labels.shape}")
        except Exception as e:
            print(f"  Error processing {rec_name}: {e}")

    return created


def print_noisy_summary_table(created: list[tuple[str, str, int, Path]]) -> None:
    """Print a summary table of all noisy segment files created."""
    if not created:
        return
    print("\nNoisy segment files created:")
    print("-" * 60)
    for record, noise_type, snr_db, path in created:
        if path.exists():
            arr = np.load(path)
            print(f"  {path.name:45} shape={arr.shape}")
    print("-" * 60)


if __name__ == "__main__":
    created = run_noise_pipeline()
    print_noisy_summary_table(created)
    print("\n" + "\033[92m" + "PHASE 1 (noise pipeline) COMPLETE" + "\033[0m")
    print(f"Total noisy segment files: {len(created)}")
