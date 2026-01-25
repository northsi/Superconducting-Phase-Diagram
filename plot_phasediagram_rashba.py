import numpy as np
import pandas as pd

colnames = [
    "v0", "v1",
    "A1g", "A2g", "Eg", "A1u", "A2u", "Eu"
]

df = pd.read_csv(
    "Tc_all_irreps.txt",
    delim_whitespace=True,
    comment="#",
    names=colnames
)


# 把 None 变成 NaN
df = df.replace("None", np.nan)

# 转成 float
for col in df.columns[2:]:
    df[col] = df[col].astype(float)

from scipy.interpolate import griddata

x = df["v0"].values
y = df["v1"].values
points = np.column_stack([x, y])

nx, ny = 400, 400   # 网格密度你可以自己调
x_fine = np.linspace(x.min(), x.max(), nx)
y_fine = np.linspace(y.min(), y.max(), ny)
Xf, Yf = np.meshgrid(x_fine, y_fine)

irreps = colnames[2:]

Tc_interp = {}
active_irreps = []   # 真正参与相图的 irreps

for ir in irreps:
    vals = df[ir].astype(float).values
    mask = ~np.isnan(vals)

    # ===== 关键判断 =====
    if np.sum(mask) < 3:
        print(f"Skip {ir}: no superconducting points")
        continue

    Tc_interp[ir] = griddata(
        points[mask],
        vals[mask],
        (Xf, Yf),
        method="linear"
    )

    active_irreps.append(ir)


# ===== 安全判定主导超导相 =====

Tc_stack = np.stack(
    [Tc_interp[ir] for ir in active_irreps],
    axis=0
)

# 哪些点所有 irreps 都是 NaN（无超导）
all_nan_mask = np.all(np.isnan(Tc_stack), axis=0)

# 只在有超导的地方算 argmax
idx = np.zeros(Tc_stack.shape[1:], dtype=int)

valid_mask = ~all_nan_mask
idx[valid_mask] = np.nanargmax(
    Tc_stack[:, valid_mask],
    axis=0
)

# 构造相图
phase_map = np.full(Xf.shape, "Normal", dtype=object)
phase_map[valid_mask] = np.array(active_irreps)[idx[valid_mask]]


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

all_phases = active_irreps + ["Normal"]
phase_to_int = {ph: i for i, ph in enumerate(all_phases)}
phase_int = np.vectorize(phase_to_int.get)(phase_map)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

cmap = ListedColormap(
    list(plt.cm.tab10.colors[:len(active_irreps)]) + [(0.85, 0.85, 0.85)]
)

plt.figure(figsize=(6, 5))

# ===== 相图 =====
mesh = plt.pcolormesh(
    Xf, Yf, phase_int,
    cmap=cmap,
    shading="auto",
    vmin=0,
    vmax=len(all_phases)
)

# ===== Tc = 10 K 等值线 =====
with np.errstate(all="ignore"):
    Tc_max = np.nanmax(Tc_stack, axis=0)

cs = plt.contour(
    Xf, Yf, Tc_max,
    levels=[10.0],
    colors="k",
    linestyles="--",
    linewidths=1.5
)
plt.clabel(cs, fmt=r"$T_c=10\,\mathrm{K}$", fontsize=9)

# ===== colorbar（离散，相名）=====
cbar = plt.colorbar(mesh, ticks=np.arange(len(all_phases)) + 0.5)
cbar.ax.set_yticklabels(all_phases)

plt.xlabel("v0")
plt.ylabel("v1")
plt.title("Superconducting Phase Diagram")

plt.tight_layout()
plt.savefig("phasediagram_with_Tc10K.png", dpi=300)
plt.close()


