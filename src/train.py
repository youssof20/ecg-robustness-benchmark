"""
Phase 2 — Train ECG arrhythmia classifiers.

Loads Phase 1 numpy arrays, trains SimpleCNN, ResNet1D, and LightweightNet with
class-weighted CE, Adam, ReduceLROnPlateau, and early stopping on validation F1.
Saves best checkpoints and training curves; reports test metrics and inference time.
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from models import get_model, _count_parameters, NUM_CLASSES

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

MODEL_NAMES = ["SimpleCNN", "ResNet1D", "LightweightNet"]
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64) + 1e-6
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def load_phase1_arrays():
    """Load train/val/test arrays from Phase 1 output directory."""
    for name in ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"]:
        path = RESULTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Run Phase 1 first. Missing: {path}")
    X_train = np.load(RESULTS_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(RESULTS_DIR / "y_train.npy").astype(np.int64)
    X_val = np.load(RESULTS_DIR / "X_val.npy").astype(np.float32)
    y_val = np.load(RESULTS_DIR / "y_val.npy").astype(np.int64)
    X_test = np.load(RESULTS_DIR / "X_test.npy").astype(np.float32)
    y_test = np.load(RESULTS_DIR / "y_test.npy").astype(np.int64)
    print(f"Loaded train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One training epoch; returns mean loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / n if n else 0.0


def evaluate(model, loader, device):
    """Return accuracy and macro F1 on the given loader."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, macro_f1, y_true, y_pred


def train_model(
    model_name: str,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    history: dict,
) -> nn.Module:
    """Train a single model; save best checkpoint; append to history."""
    print(f"\n--- Training {model_name} ---")
    model = get_model(model_name).to(DEVICE)
    # Suppress LightweightNet param print when training (already validated in get_model)
    if model_name == "LightweightNet":
        nparams = _count_parameters(model)
        print(f"  Parameters: {nparams}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE
    )
    best_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    loss_hist = []
    f1_hist = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        _, val_f1, _, _ = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_f1)
        loss_hist.append(loss)
        f1_hist.append(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_f1": val_f1},
                MODELS_DIR / f"{model_name}_best.pt",
            )
        else:
            epochs_no_improve += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: loss={loss:.4f}, val_macro_f1={val_f1:.4f}")
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    history[model_name] = {"loss": loss_hist, "f1": f1_hist}
    # Load best for return
    ckpt = torch.load(MODELS_DIR / f"{model_name}_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Best val F1: {best_f1:.4f} at epoch {best_epoch}")
    return model


def plot_training_curves(history: dict):
    """Save loss and F1 curves for all models to outputs/figures/training_curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = len(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (name, h) in enumerate(history.items()):
        epochs_range = range(1, len(h["loss"]) + 1)
        axes[0].plot(epochs_range, h["loss"], label=name)
        axes[1].plot(epochs_range, h["f1"], label=name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation macro F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved training curves to {out_path}")


def inference_time_ms(model, X_sample: torch.Tensor, n_warmup: int = 10, n_repeat: int = 100) -> float:
    """Average inference time per sample in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(X_sample)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _ = model(X_sample)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_repeat * 1000
    return elapsed


def run_training():
    """Load data, train all models, save checkpoints and curves, print test table."""
    print("Phase 2 — Training")
    print("=" * 50)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_phase1_arrays()
    class_weights = compute_class_weights(y_train)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    history = {}
    trained_models = {}
    for name in MODEL_NAMES:
        trained_models[name] = train_model(name, train_loader, val_loader, class_weights, history)

    plot_training_curves(history)

    # Test set evaluation
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    X_sample = torch.from_numpy(X_test[:1]).float().to(DEVICE)

    rows = []
    for name in MODEL_NAMES:
        model = trained_models[name].eval()
        acc, macro_f1, y_true, y_pred = evaluate(model, test_loader, DEVICE)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        nparams = _count_parameters(model)
        ms = inference_time_ms(model, X_sample)
        row = {
            "model": name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "n_params": nparams,
            "inference_ms": ms,
        }
        rows.append(row)
        print(f"\n{name}")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Per-class F1: N={per_class_f1[0]:.3f}, S={per_class_f1[1]:.3f}, V={per_class_f1[2]:.3f}, F={per_class_f1[3]:.3f}, Q={per_class_f1[4]:.3f}")
        print(f"  Parameters:  {nparams}")
        print(f"  Inference:   {ms:.2f} ms/beat")

    # Summary table
    print("\n" + "-" * 60)
    print(f"{'Model':<14} {'Accuracy':>8} {'Macro F1':>8} {'Params':>8} {'ms/beat':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['model']:<14} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} {r['n_params']:>8} {r['inference_ms']:>8.2f}")
    print("-" * 60)

    # Save CSV of test results for reproducibility
    import csv
    csv_path = RESULTS_DIR / "train_test_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "macro_f1", "f1_N", "f1_S", "f1_V", "f1_F", "f1_Q", "n_params", "inference_ms"])
        for r in rows:
            w.writerow([
                r["model"], r["accuracy"], r["macro_f1"],
                *r["per_class_f1"].tolist(), r["n_params"], r["inference_ms"],
            ])
    print(f"\nSaved test results to {csv_path}")

    print("\n" + "\033[92mPHASE 2 COMPLETE\033[0m")
    print(f"Checkpoints: {MODELS_DIR}")
    print(f"Figures:     {FIGURES_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    run_training()
