"""Fix Figure 5: reposition the native-10m reference annotation so it no
longer collides with the title; same underlying data as before, purely a
layout fix."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure titles are intentionally NOT drawn inside the image: Copernicus/NHESS
# requires the caption to be typeset text below the figure, so a title baked into
# the PNG would duplicate it. Captions live in the manuscript, not here.

_HOME = Path(os.environ["HOME"])
D_ROOT = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood"
FIG_DIR = D_ROOT / "figures"

fig, ax = plt.subplots(figsize=(8.5, 5.4))

sensors = ["NISAR\n(20 m grid)", "Sentinel-1\n(20 m grid)"]
pos_range = [(0.7, 3.8), (3.0, 11.0)]
neg_range = [(0.7, 9.0), (3.0, 11.0)]

y = [0, 1]
h = 0.32
for i, (lo, hi) in enumerate(pos_range):
    ax.barh(y[i] + h / 2, hi - lo, left=lo, height=h, color="#2e8b57",
            label=">+3 dB (candidate enhancement)" if i == 0 else None)
for i, (lo, hi) in enumerate(neg_range):
    ax.barh(y[i] - h / 2, hi - lo, left=lo, height=h, color="#c0392b",
            label="<-3 dB (candidate specular/water-like)" if i == 0 else None)

# native-10m exact reference point, placed near the NISAR row it belongs to
ax.plot([7.1], [0.16], marker="D", color="black", markersize=9, zorder=5,
        markeredgecolor="white", markeredgewidth=0.8)
ax.annotate(
    "Native 10 m reference point:\ncropland, Exp 1, NISAR\n(exact) = 7.1% both directions",
    xy=(7.1, 0.16), xytext=(7.1, 1.55), fontsize=8, ha="center", va="bottom",
    arrowprops=dict(arrowstyle="->"),
)

ax.set_yticks(y)
ax.set_yticklabels(sensors)
ax.set_xlabel("Share of vegetated-class pixels (%)")
ax.set_xlim(0, 13)
ax.set_ylim(-0.55, 2.15)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "figure5_pixel_anomaly_fractions.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Re-saved figure5_pixel_anomaly_fractions.png")
