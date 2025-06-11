import matplotlib.pyplot as plt
import numpy as np

def plot_pred_vs_true(y_true, y_pred, title="Prediction vs Truth", save_as=None):
    """
    y_true, y_pred  : 1-D numpy array (已过滤 NaN，最好是真实物理单位)
    title           : 图主标题
    save_as         : 若给文件名则保存 png；否则 plt.show()
    """
    err = y_pred - y_true

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # ① 真实 vs 预测
    ax[0].scatter(y_true, y_pred, s=12, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    ax[0].plot(lims, lims, 'k--', lw=1)          # 45° 参考线
    ax[0].set_xlabel("True I (A)")
    ax[0].set_ylabel("Pred I (A)")
    ax[0].set_title("Truth vs Pred")

    # ② 误差直方图
    ax[1].hist(err, bins=40, edgecolor='k', alpha=0.75)
    ax[1].axvline(0, color='k', ls='--', lw=1)
    ax[1].set_xlabel("Prediction Error ΔI (A)")
    ax[1].set_ylabel("Count")
    ax[1].set_title(f"MAE = {np.mean(np.abs(err)):.4f} A")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close(fig)
    else:
        plt.show()