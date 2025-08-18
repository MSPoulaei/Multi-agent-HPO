import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm

from .data import get_dataloaders
from .model.resnet9 import ResNet9
from .metrics import accuracy, f1_macro
from ..utils.io import save_json
from ..utils.plotting import plot_curves

ALLOWED_BATCHES = [32, 64, 128, 256, 512]

def validate_hparams(hp):
    # Clamp to allowed ranges and discrete options
    hp2 = {}
    hp2["optimizer"] = "adam" if hp.get("optimizer","adam").lower() not in ["adam","sgd"] else hp["optimizer"].lower()
    hp2["learning_rate"] = float(min(max(hp.get("learning_rate",1e-3), 1e-4), 1e-1))
    bs = hp.get("train_batch_size", 128)
    if bs not in ALLOWED_BATCHES:
        bs = min(ALLOWED_BATCHES, key=lambda x: abs(x - bs))
    hp2["train_batch_size"] = int(bs)
    hp2["weight_decay"] = float(min(max(hp.get("weight_decay", 5e-4), 1e-5), 1e-1))
    hp2["label_smoothing"] = float(min(max(hp.get("label_smoothing", 0.0), 0.0), 0.2))
    return hp2

def make_optimizer(model, hp):
    if hp["optimizer"] == "adam":
        return optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    else:
        return optim.SGD(model.parameters(), lr=hp["learning_rate"], momentum=0.9, nesterov=True, weight_decay=hp["weight_decay"])

def train_and_eval(
    trial_dir, hp, epochs=20, patience=8, scheduler_type="cosine",
    augment="basic", num_workers=4, amp=True, save_checkpoints=True, seed=1337, device=None
):
    os.makedirs(trial_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    hp = validate_hparams(hp)
    effective_bs = hp["train_batch_size"]
    oom_adjusted = False

    def run_training(bs):
        nonlocal oom_adjusted
        try:
            train_loader, val_loader, test_loader = get_dataloaders(
                data_root=os.path.join(trial_dir, "_data"),
                batch_size=bs, num_workers=num_workers, augment=augment, seed=seed
            )
            model = ResNet9(in_channels=3, num_classes=10).to(device)
            criterion = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
            optimizer = make_optimizer(model, hp)
            if scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            else:
                scheduler = None
            scaler = GradScaler(enabled=amp)

            best_val_acc = -1.0
            best_state = None
            wait = 0

            rows = []
            for epoch in range(1, epochs+1):
                model.train()
                tr_loss_sum = 0.0
                tr_preds, tr_targets = [], []
                for images, targets in tqdm(train_loader, desc=f"Train e{epoch}/{epochs}", leave=False):
                    images, targets = images.to(device), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=amp):
                        logits = model(images)
                        loss = criterion(logits, targets)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    tr_loss_sum += loss.item() * images.size(0)
                    preds = torch.argmax(logits.detach(), dim=1)
                    tr_preds.extend(preds.cpu().numpy())
                    tr_targets.extend(targets.cpu().numpy())

                tr_loss = tr_loss_sum / len(train_loader.dataset)
                tr_acc = accuracy(
                    preds = torch.tensor(tr_preds).numpy(),
                    targets = torch.tensor(tr_targets).numpy()
                )
                tr_f1 = f1_macro(tr_targets, tr_preds)

                # Validation
                model.eval()
                va_loss_sum = 0.0
                va_preds, va_targets = [], []
                with torch.no_grad():
                    for images, targets in tqdm(val_loader, desc=f"Val e{epoch}/{epochs}", leave=False):
                        images, targets = images.to(device), targets.to(device)
                        with autocast(enabled=amp):
                            logits = model(images)
                            loss = criterion(logits, targets)
                        va_loss_sum += loss.item() * images.size(0)
                        preds = torch.argmax(logits, dim=1)
                        va_preds.extend(preds.cpu().numpy())
                        va_targets.extend(targets.cpu().numpy())
                va_loss = va_loss_sum / len(val_loader.dataset)
                va_acc = accuracy(
                    preds = torch.tensor(va_preds).numpy(),
                    targets = torch.tensor(va_targets).numpy()
                )
                from sklearn.metrics import f1_score
                va_f1 = f1_score(va_targets, va_preds, average="macro")

                if scheduler is not None:
                    scheduler.step()

                rows.append({
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "train_acc": tr_acc,
                    "val_acc": va_acc,
                    "train_f1": tr_f1,
                    "val_f1": va_f1,
                    "lr": optimizer.param_groups[0]["lr"],
                })

                # Early stopping on val_acc
                if va_acc > best_val_acc:
                    best_val_acc = va_acc
                    best_state = {
                        "model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch
                    }
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            metrics_df = pd.DataFrame(rows)
            metrics_path = os.path.join(trial_dir, "metrics_epoch.csv")
            metrics_df.to_csv(metrics_path, index=False)

            # Save plots
            from ..utils.plotting import plot_curves
            plot_curves(metrics_df, trial_dir)

            # Restore best and test
            if best_state is not None:
                model.load_state_dict(best_state["model"])
            if save_checkpoints and best_state is not None:
                torch.save(best_state, os.path.join(trial_dir, "checkpoint_best.pt"))

            # Test
            model.eval()
            te_loss_sum = 0.0
            te_preds, te_targets = [], []
            with torch.no_grad():
                for images, targets in tqdm(test_loader, desc=f"Test", leave=False):
                    images, targets = images.to(device), targets.to(device)
                    with autocast(enabled=amp):
                        logits = model(images)
                        loss = criterion(logits, targets)
                    te_loss_sum += loss.item() * images.size(0)
                    preds = torch.argmax(logits, dim=1)
                    te_preds.extend(preds.cpu().numpy())
                    te_targets.extend(targets.cpu().numpy())
            from sklearn.metrics import f1_score
            te_loss = te_loss_sum / len(test_loader.dataset)
            te_acc = accuracy(torch.tensor(te_preds).numpy(), torch.tensor(te_targets).numpy())
            te_f1 = f1_score(te_targets, te_preds, average="macro")

            summary = {
                "effective_hparams": dict(hp, train_batch_size=bs),
                "best_val_acc": float(best_val_acc),
                "test_loss": float(te_loss),
                "test_acc": float(te_acc),
                "test_f1": float(te_f1),
                "oom_adjusted": bool(oom_adjusted),
            }
            return summary, metrics_df

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return None, None
            else:
                raise

    # Attempt with desired batch size; back off if OOM
    idx = ALLOWED_BATCHES.index(effective_bs) if effective_bs in ALLOWED_BATCHES else 2
    while idx >= 0:
        result = run_training(ALLOWED_BATCHES[idx])
        if result[0] is not None:
            if ALLOWED_BATCHES[idx] != effective_bs:
                oom_adjusted = True
                result[0]["oom_adjusted"] = True
            break
        idx -= 1

    if result[0] is None:
        raise RuntimeError("OOM even at smallest batch size")

    # Save summary JSON
    save_json(os.path.join(trial_dir, "summary.json"), result[0])
    return result