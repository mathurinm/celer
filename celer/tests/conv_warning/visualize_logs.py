import re
import numpy as np
import matplotlib.pyplot as plt


def parse_logs(filename: str) -> np.ndarray:
    with open(filename) as f:
        logs = f.readlines()

    li_primal_gap = []
    # loop over even lines
    for i in range(0, len(logs), 2):
        # clean line
        pattern = r"Iter|:|primal|gap|\n|,"
        log = re.sub(pattern, "", logs[i])

        # parse logs
        _, primal, gap = log.split()
        primal, gap = float(primal), float(gap)

        li_primal_gap.append([primal, gap])

    arr = np.array(li_primal_gap, dtype=float)
    return arr


def visualize_logs(arr_primal_gap: np.ndarray, title: str = "") -> None:
    title_vis = f"LogReg primal-gap {title}"
    primal = arr_primal_gap[:, 0]
    gap = arr_primal_gap[:, 1]

    # plot
    primal_residuals = primal - primal.min()
    fig, ax1 = plt.subplots()
    ax1.semilogy(primal_residuals, label="primal residual", color='red')
    ax1.set(xlabel='iteration', ylabel='primal residual')

    ax2 = ax1.twinx()
    ax2.semilogy(gap, label="gap", color='blue')
    ax2.set(ylabel='gap')

    # set layout
    ax1.set_title(title_vis)
    fig.legend()
    fig.tight_layout()

    plt.show()
    return


# process logs
filename_sk = 'celer/tests/conv_warning/logs/sklearn-check.txt'
arr_sk = parse_logs(filename_sk)

visualize_logs(
    arr_primal_gap=arr_sk,
    title="sklearn"
)

# process logs
filename_build = 'celer/tests/conv_warning/logs/build-dataset.txt'
arr_build = parse_logs(filename_build)

visualize_logs(
    arr_primal_gap=arr_build,
    title="build dataset"
)
