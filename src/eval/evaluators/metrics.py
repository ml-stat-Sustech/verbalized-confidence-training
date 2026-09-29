import matplotlib.pyplot as plt
import numpy as np


def compute_pass_n(is_correct, k):
    n = len(is_correct[0])
    corrects, totals = [], []
    for correctness_list in is_correct:
        count = 0
        for j in range(n):
            if correctness_list[j] == 1:
                count += 1
        corrects.append(count)
        totals.append(n)
    return estimate_pass_at_k(totals, corrects, k).mean()


def estimate_pass_at_k(num_samples, num_correct, k):
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def plot_reliability_diagram(correctness, confidence, n_bins=15, title=None, save_path=None):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(n_bins):
        in_bin = bin_indices == i
        if np.sum(in_bin) > 0:
            accuracy = np.mean(correctness[in_bin])
            mean_confidence = np.mean(confidence[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)

    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)
    weights = np.histogram(confidence, bins)[0] / len(confidence)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc))

    delta = 1.0 / n_bins
    x = np.arange(0, 1, delta)
    mid = np.linspace(delta / 2, 1 - delta / 2, n_bins)
    error = np.abs(np.subtract(mid, bin_acc))

    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(6.7, 6))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(color="tab:grey", linestyle=(0, (1, 5)), linewidth=1, zorder=0)
    plt.bar(x, bin_acc, color="#6B9EFF", width=delta, align="edge", edgecolor="k", label="Outputs", zorder=5)
    plt.bar(
        x,
        error,
        bottom=np.minimum(bin_acc, mid),
        color="mistyrose",
        alpha=0.5,
        width=delta,
        align="edge",
        edgecolor="r",
        hatch="/",
        label="Gap",
        zorder=10,
    )
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="tab:grey", zorder=15)
    plt.ylabel("Accuracy", fontsize=24)
    plt.xlabel("Confidence", fontsize=24)
    plt.legend(loc="upper left", framealpha=1.0, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.text(
        0.47,
        0.065,
        f"ECE: {ece * 100:.2f}%",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round, pad=0.5", facecolor="#FFDAB9", edgecolor="#D2691E"),
        fontsize=28,
        color="darkblue",
        zorder=20,
    )

    if title is not None:
        plt.title(title, fontsize=36)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plt
