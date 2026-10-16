"""
Manuscript Figures 2-7, built from exact numbers already published in
manuscript-draft.md (Tables 2, 4, 5, 6, 7, 9, 10) and the NASA POWER
rainfall series recorded in phase3-preliminary-differencing.md. No
hand-estimated or invented per-class numbers are introduced anywhere in
this script; wherever the manuscript itself reports only a range (Uttar
Khan Experiment 2 Sentinel-1; the Figure 5 pixel-anomaly fractions), the
figure shows that same range rather than fabricating a precise value.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# Figure titles are intentionally NOT drawn inside the image: Copernicus/NHESS
# requires the caption to be typeset text below the figure, so a title baked into
# the PNG would duplicate it. Captions live in the manuscript, not here.

_HOME = Path(os.environ["HOME"])
D_ROOT = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood"
FIG_DIR = D_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9})

# ---------------------------------------------------------------------------
# Figure 2: ESA WorldCover class-mix, Demra (full) and Demra-vs-Uttar Khan
# ---------------------------------------------------------------------------
demra_classes = ["Built-up", "Cropland", "Tree cover", "Bare/sparse",
                  "Permanent water", "Grassland", "Wetland", "Shrubland"]
demra_pct = [37.2, 29.4, 15.9, 10.6, 5.5, 0.7, 0.6, 0.06]

sites_b = ["Demra", "Uttar Khan"]
built_up_b = [37.2, 24.4]
treecrop_b = [45.3, 69.1]
other_b = [100 - 37.2 - 45.3, 100 - 24.4 - 69.1]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

ax = axes[0]
order = np.argsort(demra_pct)[::-1]
bars = ax.bar([demra_classes[i] for i in order], [demra_pct[i] for i in order],
              color="#3b7dd8")
ax.set_ylabel("Share of ward area (%)")
ax.set_title("A. Demra land-cover composition\n(ESA WorldCover 10 m, 2020)")
ax.tick_params(axis="x", rotation=40)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:g}%", (b.get_x() + b.get_width() / 2, h), ha="center",
                va="bottom", fontsize=7)

ax = axes[1]
x = np.arange(len(sites_b))
w = 0.6
b1 = ax.bar(x, built_up_b, w, label="Built-up", color="#fa4040")
b2 = ax.bar(x, treecrop_b, w, bottom=built_up_b, label="Tree cover + cropland", color="#2e8b57")
bottom2 = [built_up_b[i] + treecrop_b[i] for i in range(2)]
b3 = ax.bar(x, other_b, w, bottom=bottom2, label="Other (grassland, wetland,\nbare/sparse, water, shrubland)", color="#b4b4b4")
ax.set_xticks(x)
ax.set_xticklabels(sites_b)
ax.set_ylabel("Share of site area (%)")
ax.set_title("B. Vegetated-land share by site")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
for i in range(2):
    ax.annotate(f"{built_up_b[i]:g}%", (i, built_up_b[i] / 2), ha="center", va="center", fontsize=7, color="white")
    ax.annotate(f"{treecrop_b[i]:g}%", (i, built_up_b[i] + treecrop_b[i] / 2), ha="center", va="center", fontsize=7, color="white")
    ax.annotate(f"{other_b[i]:.1f}%", (i, bottom2[i] + other_b[i] / 2), ha="center", va="center", fontsize=7)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG_DIR / "figure2_worldcover_composition.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figure2_worldcover_composition.png")

# ---------------------------------------------------------------------------
# Figure 3: NASA POWER daily rainfall at Demra, 1-25 July 2026
# ---------------------------------------------------------------------------
days = list(range(1, 26))
rain = [8.6, 8.8, 1.7, 17.1, 8.5, 12.7, 35.2, 53.7, 35.4, 59.8, 43.5, 41.9,
        39.8, 14.4, 4.7, 12.4, 12.8, 9.3, 12.0, 25.5, 30.4, 21.4, 17.6, 11.7, 4.6]
assert len(days) == len(rain) == 25

# acquisitions falling inside the plotted 1-25 July window
acq = [
    (2, "NISAR pre\n(Exp 2)", "tab:blue", "o"),
    (1, "S1 pre\n(Exp 1)", "tab:orange", "s"),
    (3, "S1 pre\n(Exp 2)", "tab:orange", "s"),
    (12, "NISAR during\n(Exp 1)", "tab:blue", "o"),
    (13, "S1 during\n(Exp 1)", "tab:orange", "s"),
    (14, "NISAR during\n(Exp 2)", "tab:blue", "o"),
    (15, "S1 during\n(Exp 2)", "tab:orange", "s"),
]

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(days, rain, color="#4c86c4", width=0.8, zorder=2)
ax.axvspan(7, 13, color="orange", alpha=0.12, zorder=1, label="7-day deluge (7-13 Jul, ~309 mm total)")

for day, label, color, marker in acq:
    ax.plot(day, rain[day - 1] + 4, marker=marker, color=color, markersize=8,
            markeredgecolor="black", markeredgewidth=0.6, zorder=3)
    ax.annotate(label, (day, rain[day - 1] + 6), ha="center", fontsize=6.5,
                rotation=0, va="bottom")

ax.annotate("NISAR pre (Exp 1),\n30 Jun 2026\n(one day before\nplotted window)",
            xy=(1, 8.6), xytext=(-2.3, 45), fontsize=6.5, ha="center",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))

ax.set_xlim(0.2, 25.8)
ax.set_xlabel("July 2026")
ax.set_ylabel("Daily precipitation (mm)\n(NASA POWER, PRECTOTCORR)")
ax.set_xticks(days)
handles = [
    Patch(facecolor="#4c86c4", label="Daily rainfall"),
    Patch(facecolor="orange", alpha=0.12, label="7-day deluge, 7-13 Jul (~309 mm total)"),
    plt.Line2D([0], [0], marker="o", color="tab:blue", linestyle="", markeredgecolor="black", label="NISAR acquisition"),
    plt.Line2D([0], [0], marker="s", color="tab:orange", linestyle="", markeredgecolor="black", label="Sentinel-1 acquisition"),
]
ax.legend(handles=handles, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIG_DIR / "figure3_rainfall_timeseries.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figure3_rainfall_timeseries.png")

# ---------------------------------------------------------------------------
# Figure 4: delta heatmap, class x sensor x experiment x site (Demra + Uttar Khan)
# ---------------------------------------------------------------------------
classes4 = ["Tree cover", "Shrubland", "Grassland", "Cropland", "Wetland"]

# Table 4 -- Demra
demra_exp1_nisar = [0.09, -0.23, 0.12, 0.02, 0.07]
demra_exp1_s1    = [0.34, -0.40, 0.17, -0.16, -0.12]
demra_exp2_nisar = [0.56, -0.42, 0.01, -0.23, -0.86]
demra_exp2_s1    = [0.33, -0.12, 0.50, 0.01, 0.29]

# Table 5 -- Uttar Khan, Experiment 1 only (Exp 2 NISAR is the excluded artifact,
# Exp 2 S1 is reported only as a range, never itemized per class in the manuscript)
uk_exp1_nisar = [0.54, 0.26, 0.23, -0.22, -0.11]
uk_exp1_s1    = [0.42, -0.13, -0.14, -0.77, -0.83]

cols = [
    ("Demra\nExp1\nNISAR", demra_exp1_nisar, False, None),
    ("Demra\nExp1\nS1", demra_exp1_s1, False, None),
    ("Demra\nExp2\nNISAR", demra_exp2_nisar, False, None),
    ("Demra\nExp2\nS1", demra_exp2_s1, False, None),
    ("Uttar Khan\nExp1\nNISAR", uk_exp1_nisar, False, None),
    ("Uttar Khan\nExp1\nS1", uk_exp1_s1, False, None),
    ("Uttar Khan\nExp2\nNISAR", [np.nan] * 5, True, "Excluded:\nacquisition-mode\nartifact\n(Table 6, Fig. 7)"),
    ("Uttar Khan\nExp2\nS1", [np.nan] * 5, True, "Not itemized;\nreported only as\n-0.36 to -1.47 dB\nrange (all classes)"),
]

data = np.array([c[1] for c in cols]).T  # rows=classes, cols=experiments
vmax = 1.0
fig, ax = plt.subplots(figsize=(11, 5.2))
masked = np.ma.masked_invalid(data)
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad(color="#e0e0e0")
im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

ax.set_xticks(range(len(cols)))
ax.set_xticklabels([c[0] for c in cols], fontsize=7.5)
ax.set_yticks(range(len(classes4)))
ax.set_yticklabels(classes4)

for i in range(len(classes4)):
    for j, c in enumerate(cols):
        if c[2]:
            continue
        val = data[i, j]
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(val) > 0.55 else "black")

# hatch the two excluded/non-itemized columns
for j, c in enumerate(cols):
    if c[2]:
        ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(classes4), fill=False,
                                hatch="//", edgecolor="black", linewidth=0.8))
        ax.text(j, (len(classes4) - 1) / 2, c[3], ha="center", va="center",
                fontsize=6.5, style="italic")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Mean Δ backscatter, during - pre (dB)")
fig.tight_layout()
fig.savefig(FIG_DIR / "figure4_delta_heatmap.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figure4_delta_heatmap.png")

# ---------------------------------------------------------------------------
# Figure 5: pixel-level anomaly fractions, Experiment 1, Demra
# reported in the manuscript ONLY as per-sensor ranges across the five
# vegetated classes, plus one exact native-10m reference point (cropland)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))

sensors = ["NISAR\n(20 m grid)", "Sentinel-1\n(20 m grid)"]
pos_range = [(0.7, 3.8), (3.0, 11.0)]
neg_range = [(0.7, 9.0), (3.0, 11.0)]

y = np.arange(len(sensors))
h = 0.32
for i, (lo, hi) in enumerate(pos_range):
    ax.barh(y[i] + h / 2, hi - lo, left=lo, height=h, color="#2e8b57", label=">+3 dB (candidate enhancement)" if i == 0 else None)
for i, (lo, hi) in enumerate(neg_range):
    ax.barh(y[i] - h / 2, hi - lo, left=lo, height=h, color="#c0392b", label="<-3 dB (candidate specular/water-like)" if i == 0 else None)

ax.plot([7.1], [1.6], marker="D", color="black", markersize=9, zorder=5)
ax.annotate("Native 10 m reference point:\ncropland, Exp 1, NISAR\n(exact) = 7.1% both directions", (7.1, 1.6),
            xytext=(9.5, 1.75), fontsize=8, arrowprops=dict(arrowstyle="->"))
ax.plot([7.1], [1.6], marker="D", color="none")  # keep scale sane

ax.set_yticks(y)
ax.set_yticklabels(sensors)
ax.set_xlabel("Share of vegetated-class pixels (%)")
ax.set_xlim(0, 13)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "figure5_pixel_anomaly_fractions.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figure5_pixel_anomaly_fractions.png")

# ---------------------------------------------------------------------------
# Figure 6: native-10m vs 20m robustness check (Experiment 1, NISAR, Demra)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

classes6 = ["Tree cover", "Cropland", "Wetland"]
delta_20m = [0.09, 0.02, 0.07]
delta_10m = [0.10, 0.02, 0.04]

ax = axes[0]
x = np.arange(len(classes6))
w = 0.35
ax.bar(x - w / 2, delta_20m, w, label="20 m grid (primary)", color="#3b7dd8")
ax.bar(x + w / 2, delta_10m, w, label="Native 10 m (robustness check)", color="#f0a830")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(classes6)
ax.set_ylabel("Mean Δ backscatter (dB)")
ax.set_title("A. Class-mean Δ, Experiment 1 NISAR, Demra")
ax.legend(fontsize=7.5)
for i in range(len(classes6)):
    ax.annotate(f"{delta_20m[i]:+.2f}", (x[i] - w / 2, delta_20m[i]), ha="center",
                va="bottom" if delta_20m[i] >= 0 else "top", fontsize=7)
    ax.annotate(f"{delta_10m[i]:+.2f}", (x[i] + w / 2, delta_10m[i]), ha="center",
                va="bottom" if delta_10m[i] >= 0 else "top", fontsize=7)

ax = axes[1]
pix_20m = 121688
pix_10m_approx = pix_20m * 4  # manuscript: "roughly four times the pixel count"
bars = ax.bar(["20 m grid\n(primary)", "Native 10 m\n(~4x, approx.)"], [pix_20m, pix_10m_approx],
              color=["#3b7dd8", "#f0a830"])
ax.set_ylabel("Analysis-grid pixel count, Demra")
ax.set_title("B. Total pixel count, 20 m vs native 10 m")
for b, v in zip(bars, [pix_20m, pix_10m_approx]):
    ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIG_DIR / "figure6_native10m_robustness.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figure6_native10m_robustness.png")

# ---------------------------------------------------------------------------
# Figure 7: the calibration-artifact figure, 4 panels (A/B original + new C/D)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))

# Panel A -- Table 6, permanent-water shift, mode-matched vs mode-mismatched
ax = axes[0, 0]
labels_a = ["Demra\n(2 Jul->14 Jul,\nmode-mismatched)", "Uttar Khan\n(2 Jul->14 Jul,\nmode-mismatched)", "Uttar Khan\n(30 Jun->12 Jul,\nmode-matched)"]
vals_a = [-0.65, -8.00, 0.15]
colors_a = ["#888888", "#c0392b", "#2e8b57"]
bars = ax.bar(labels_a, vals_a, color=colors_a)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Permanent-water Δ backscatter (dB)")
ax.set_title("A. NISAR permanent-water shift:\nmode-matched vs mode-mismatched scene pairs")
for b, v in zip(bars, vals_a):
    ax.annotate(f"{v:+.2f} dB", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax.tick_params(axis="x", labelsize=7.5)

# Panel B -- observed shift vs ALOS-2/PALSAR-2 precedent
ax = axes[0, 1]
cats_b = ["Uttar Khan\nartifact\n(this study)", "ALOS-2/PALSAR-2\npre-recalibration\n(Motohka et al. 2018)", "ALOS-2/PALSAR-2\npost-recalibration\n(Motohka et al. 2018)"]
low_b = [8.00, 0.231, 0.0]
high_b = [8.00, 2.188, 0.8]
x = np.arange(len(cats_b))
for i in range(len(cats_b)):
    if low_b[i] == high_b[i]:
        ax.bar(x[i], high_b[i], color="#c0392b", width=0.5)
        ax.annotate(f"{high_b[i]:.2f} dB", (x[i], high_b[i]), ha="center", va="bottom", fontsize=8)
    else:
        ax.bar(x[i], high_b[i] - low_b[i], bottom=low_b[i], color="#888888", width=0.5)
        ax.annotate(f"{low_b[i]:.3f}-{high_b[i]:.3f} dB", (x[i], high_b[i]), ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(cats_b, fontsize=7.5)
ax.set_ylabel("Magnitude of mode-related radiometric shift (dB)")
ax.set_title("B. Observed shift vs closest documented precedent")

# Panel C (new) -- Hakaluki Haor co-pol permanent-water calibration check
ax = axes[1, 0]
labels_c = ["NISAR HHHH", "Sentinel-1 VV"]
vals_c = [0.12, 1.94]
colors_c = ["#2e8b57", "#c0392b"]
bars = ax.bar(labels_c, vals_c, color=colors_c)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Permanent-water Δ backscatter (dB)")
ax.set_title("C. Hakaluki Haor co-pol calibration check\n(Table 9; n=75,442 water pixels)")
for b, v in zip(bars, vals_c):
    ax.annotate(f"{v:+.2f} dB", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 2.4)

# Panel D (new) -- Kaptai Lake permanent-water calibration check, 3 channels
ax = axes[1, 1]
labels_d = ["NISAR\nHVHV", "Sentinel-1\nVH", "Sentinel-1\nVV"]
vals_d = [-0.46, -1.79, -2.71]
colors_d = ["#2e8b57", "#c0392b", "#c0392b"]
bars = ax.bar(labels_d, vals_d, color=colors_d)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Permanent-water Δ backscatter (dB)")
ax.set_title("D. Kaptai Lake calibration check, all channels\n(Table 10; n=503,242-740,912 water pixels)")
for b, v in zip(bars, vals_d):
    ax.annotate(f"{v:+.2f} dB", (b.get_x() + b.get_width() / 2, v), ha="center", va="top", fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / "figure7_calibration_artifact.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figure7_calibration_artifact.png")

print("\nAll figures 2-7 saved to", FIG_DIR)
