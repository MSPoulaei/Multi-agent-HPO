import os
import matplotlib.pyplot as plt

def plot_curves(metrics_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = metrics_df["epoch"]
    
    # Loss
    plt.figure()
    plt.plot(epochs, metrics_df["train_loss"], label="train_loss")
    plt.plot(epochs, metrics_df["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    # Val loss alone
    plt.figure()
    plt.plot(epochs, metrics_df["val_loss"], label="val_loss", color="orange")
    plt.xlabel("epoch"); plt.ylabel("val_loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "val_loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, metrics_df["train_acc"], label="train_acc")
    plt.plot(epochs, metrics_df["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "acc.png"))
    plt.close()

    # F1
    plt.figure()
    plt.plot(epochs, metrics_df["train_f1"], label="train_f1")
    plt.plot(epochs, metrics_df["val_f1"], label="val_f1")
    plt.xlabel("epoch"); plt.ylabel("f1_macro"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "f1.png"))
    plt.close()

    # Combined
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0,0].plot(epochs, metrics_df["train_loss"], label="train_loss")
    axs[0,0].plot(epochs, metrics_df["val_loss"], label="val_loss")
    axs[0,0].set_title("Loss"); axs[0,0].legend(); axs[0,0].grid(True)

    axs[0,1].plot(epochs, metrics_df["val_loss"], label="val_loss", color="orange")
    axs[0,1].set_title("Val Loss"); axs[0,1].legend(); axs[0,1].grid(True)

    axs[1,0].plot(epochs, metrics_df["train_acc"], label="train_acc")
    axs[1,0].plot(epochs, metrics_df["val_acc"], label="val_acc")
    axs[1,0].set_title("Accuracy"); axs[1,0].legend(); axs[1,0].grid(True)

    axs[1,1].plot(epochs, metrics_df["train_f1"], label="train_f1")
    axs[1,1].plot(epochs, metrics_df["val_f1"], label="val_f1")
    axs[1,1].set_title("F1 macro"); axs[1,1].legend(); axs[1,1].grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "combined.png"))
    plt.close(fig)