"""Hàm vẽ biểu đồ dùng chung (matplotlib, xuất PNG cho báo cáo/slide).

Bảng màu phân loại theo thứ tự cố định (không xoay vòng), thang tuần tự một
màu (xanh) cho heatmap, lưới/trục mờ, không dùng trục kép.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENIGN, ATTACK = SERIES[0], SERIES[1]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def setup():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "font.size": 10,
            "lines.linewidth": 2,
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        }
    )


setup()


def save(fig, name: str):
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def heatmap(ax, data, row_labels, col_labels, fmt="{:.2f}", vmin=0, vmax=1, cmap=SEQ):
    data = np.asarray(data, dtype=float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                txt, color = "–", TEXT_SECONDARY
            else:
                txt = fmt.format(v)
                frac = (v - vmin) / (vmax - vmin + 1e-12)
                color = "white" if frac > 0.55 else TEXT_PRIMARY
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    return im


def barh(ax, labels, values, color=SERIES[0], fmt="{:.3f}", err=None):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.6, xerr=err, error_kw={"ecolor": TEXT_SECONDARY, "lw": 1})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9, color=TEXT_SECONDARY)
