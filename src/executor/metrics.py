import numpy as np
from sklearn.metrics import f1_score

def accuracy(preds, targets):
    return (preds == targets).mean()

def f1_macro(y_true, y_pred, num_classes=10):
    return f1_score(y_true, y_pred, average="macro")

def trend_features(series):
    x = np.arange(len(series))
    y = np.array(series, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    last = y[-1]
    first = y[0]
    delta = last - first
    min_val = y.min()
    max_val = y.max()
    return {"slope": float(slope), "delta": float(delta), "first": float(first), "last": float(last), "min": float(min_val), "max": float(max_val)}

def heuristic_analysis(metrics_df):
    # Basic signals
    tr_loss = metrics_df["train_loss"].values
    va_loss = metrics_df["val_loss"].values
    tr_acc = metrics_df["train_acc"].values
    va_acc = metrics_df["val_acc"].values
    tr_f1  = metrics_df["train_f1"].values
    va_f1  = metrics_df["val_f1"].values

    gaps_acc = tr_acc - va_acc
    gaps_f1  = tr_f1 - va_f1

    overfit = (gaps_acc[-1] > 0.15) and (va_loss[-1] > va_loss.min()*1.02)
    underfit = (tr_acc[-1] < 0.6) and (va_acc[-1] < 0.55)
    lr_too_high = (va_loss[0] < va_loss[1]*0.8) or np.any(np.diff(va_loss) > 0.5)  # spikes
    plateau = (np.mean(np.abs(np.diff(va_acc[-5:]))) < 0.002) and (va_acc[-1] < 0.8)
    lr_too_low = (va_loss[0] - va_loss[5] < 0.05) if len(va_loss) > 5 else False
    noisy = (np.std(np.diff(va_loss)) > 0.2)

    # crude label smoothing detection: if smoothing high and train accuracy suppressed vs val
    # actual smoothing known in training loop; here infer via pattern
    smoothing_suspect = (tr_acc[-1] + 0.03 < va_acc[-1]) and (tr_acc[-1] < 0.75)

    features = {
        "train_loss_trend": trend_features(tr_loss),
        "val_loss_trend": trend_features(va_loss),
        "train_acc_trend": trend_features(tr_acc),
        "val_acc_trend": trend_features(va_acc),
        "train_f1_trend": trend_features(tr_f1),
        "val_f1_trend": trend_features(va_f1),
        "final_gaps": {"acc_gap": float(gaps_acc[-1]), "f1_gap": float(gaps_f1[-1])},
        "flags": {
            "overfitting": bool(overfit),
            "underfitting": bool(underfit),
            "lr_too_high": bool(lr_too_high),
            "lr_too_low": bool(lr_too_low),
            "plateau": bool(plateau),
            "noisy_updates": bool(noisy),
            "smoothing_suspect": bool(smoothing_suspect),
        }
    }
    keywords = [k for k, v in features["flags"].items() if v]
    if len(keywords) == 0:
        keywords = ["healthy_training"]
    return features, keywords