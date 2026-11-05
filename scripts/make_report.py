"""Builds LaTeX tables, figures and number macros for the paper from results/*.json."""
import json, glob, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description="Build LaTeX tables/figures from results/*.json")
ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper_assets")
args = ap.parse_args()
RES, OUT = args.results, args.out
os.makedirs(f"{OUT}/tables", exist_ok=True)
os.makedirs(f"{OUT}/figs", exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "font.family": "serif",
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})

R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{RES}/*.json")}
R = {k: v for k, v in R.items() if "test" in v}
NAMES = {"oracle": "Oracle (all polygon)$^\\dagger$", "polyonly": "Poly-only", "partial": "Partial-FedAvg",
         "masked": "Masked-FedAvg", "boxaspoly": "Box-as-Polygon", "fedcobra": "\\textbf{FedCoBRA (ours)}"}
ORDER = ["oracle", "polyonly", "partial", "masked", "boxaspoly", "fedcobra"]


def get(tag, split, key):
    return R[tag][split][key] if tag in R else None


def agg(tags, split, key):
    v = [get(t, split, key) for t in tags if t in R]
    if not v:
        return None, None, 0
    v = np.array(v) * 100
    return v.mean(), (v.max() - v.min()) / 2, len(v)


def fmt(m, h, n, bold=False):
    if m is None:
        return "--"
    s = f"{m:.1f}" + (f"$\\pm${h:.1f}" if n > 1 else "")
    return f"\\textbf{{{s}}}" if bold else s


macros = {}

# ---------------------------------------------------------------- main table
cols = [("test", "cmiou"), ("test", "miou"), ("test", "r75"), ("test", "box_iou"),
        ("unseen", "cmiou"), ("unseen", "miou")]
rows = []
best = {}
for c in cols:
    vals = {m: agg([f"main_s{s}_{m}" for s in (0, 1)], *c)[0] for m in ORDER if m != "oracle"}
    vals = {k: v for k, v in vals.items() if v is not None}
    best[c] = max(vals, key=vals.get) if vals else None
nseeds = 0
for m in ORDER:
    cells = []
    for c in cols:
        mu, h, n = agg([f"main_s{s}_{m}" for s in (0, 1)], *c)
        nseeds = max(nseeds, n)
        cells.append(fmt(mu, h, n, bold=(best[c] == m)))
        if mu is not None:
            macros[f"main{m}{c[0]}{c[1]}"] = f"{mu:.1f}"
    nm = NAMES[m] + ("$^\\ast$" if sum(f"main_s{s}_{m}" in R for s in (0, 1)) == 1 else "")
    rows.append(f"{nm} & " + " & ".join(cells) + " \\\\" + ("\\midrule" if m == "oracle" else ""))
tex = r"""\begin{table*}[t]
\centering
\caption{Main results on SynPoly-FL with 3 polygon and 7 box clients ($\rho=0.3$). Mean $\pm$ half-range over %d seed(s); all numbers in \%%. cmIoU: mask IoU of the contour predicted from the true object centre (contour quality); mIoU, R@0.75: end-to-end (predicted centre). Best non-oracle result in bold; $^\ast$single seed (seed 0). $^\dagger$Oracle labels all 10 clients with polygons ($3.3\times$ more polygon annotation).}
\label{tab:main}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
& \multicolumn{4}{c}{Seen domains (600 test images)} & \multicolumn{2}{c}{Unseen domains (300)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
Method & cmIoU & mIoU & R@0.75 & Box IoU & cmIoU & mIoU \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (nseeds, "\n".join(rows))
open(f"{OUT}/tables/main.tex", "w").write(tex)

# ---------------------------------------------------------------- ablation table
abl = [("main_s0_fedcobra", "FedCoBRA (full)"), ("abl_s0_noRADA", "w/o RADA (FedAvg for $\\theta_C$)"),
       ("abl_s0_noShape", "w/o shape prior"), ("abl_s0_noDistill", "w/o teacher distillation"),
       ("abl_s0_noAff", "w/o affinity loss")]
rows = []
for tag, name in abl:
    cells = [fmt(*agg([tag], "test", "cmiou")), fmt(*agg([tag], "test", "miou")),
             fmt(*agg([tag], "unseen", "cmiou")), fmt(*agg([tag], "unseen", "miou"))]
    rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    if tag in R:
        macros["abl" + tag.split("_")[-1] + "cmiou"] = f"{get(tag, 'test', 'cmiou') * 100:.1f}"
        macros["abl" + tag.split("_")[-1] + "ucmiou"] = f"{get(tag, 'unseen', 'cmiou') * 100:.1f}"
tex = r"""\begin{table}[t]
\centering
\caption{Ablation of FedCoBRA (seed 0, $\rho=0.3$; \%%).}
\label{tab:ablation}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
& \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Variant & cmIoU & mIoU & cmIoU & mIoU \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows)
open(f"{OUT}/tables/ablation.tex", "w").write(tex)

# ---------------------------------------------------------------- budget table
bud = [("allpoly", "All polygon (FedAvg)", "60 P"), ("allbox", "All box (FedCoBRA losses)", "679 B"),
       ("mixed", "Mixed 3P+7B (FedCoBRA)", "60/679")]
rows = []
for key, name, n in bud:
    tags = [f"budget_s{s}_{key}" for s in (0, 1)]
    cells = [fmt(*agg(tags, "test", "cmiou")), fmt(*agg(tags, "test", "miou")),
             fmt(*agg(tags, "unseen", "cmiou")), fmt(*agg(tags, "unseen", "miou"))]
    rows.append(f"{name} & {n} & " + " & ".join(cells) + " \\\\")
    mu = agg(tags, "test", "cmiou")[0]
    if mu is not None:
        macros["bud" + key + "cmiou"] = f"{mu:.1f}"
        macros["bud" + key + "ucmiou"] = f"{agg(tags, 'unseen', 'cmiou')[0]:.1f}"
        macros["bud" + key + "miou"] = f"{agg(tags, 'test', 'miou')[0]:.1f}"
tex = r"""\begin{table}[t]
\centering
\caption{Equal annotation budget per client (time of 60 polygons, i.e.\ 79.2\,s$\times$60; boxes by extreme clicking at 7\,s give 679 images). Seed 0 (\%%).}
\label{tab:budget}
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
& Images & \multicolumn{2}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Annotation strategy & /client & cmIoU & mIoU & cmIoU & mIoU \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows)
open(f"{OUT}/tables/budget.tex", "w").write(tex)

# ---------------------------------------------------------------- optimiser table
opt = [("main_s0_oracle", "FedAvg"), ("opt_s0_fedprox", "FedProx ($\\mu=0.01$)")]
rows = []
for tag, name in opt:
    cells = [fmt(*agg([tag], "test", "cmiou")), fmt(*agg([tag], "test", "miou")), fmt(*agg([tag], "test", "box_iou")),
             fmt(*agg([tag], "unseen", "cmiou")), fmt(*agg([tag], "unseen", "miou"))]
    rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    if tag in R:
        macros["opt" + tag.split("_")[-1] + "cmiou"] = f"{get(tag, 'test', 'cmiou') * 100:.1f}"
        macros["opt" + tag.split("_")[-1] + "miou"] = f"{get(tag, 'test', 'miou') * 100:.1f}"
tex = r"""\begin{table}[t]
\centering
\caption{Federated optimisers with polygon labels on all clients (seed 0, \%%).}
\label{tab:optim}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
& \multicolumn{3}{c}{Seen} & \multicolumn{2}{c}{Unseen} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Optimiser & cmIoU & mIoU & Box IoU & cmIoU & mIoU \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows)
open(f"{OUT}/tables/optim.tex", "w").write(tex)

# ---------------------------------------------------------------- figure: curves + reliability
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.45))
colors = {"oracle": "k", "polyonly": "tab:gray", "partial": "tab:purple", "masked": "tab:blue",
          "boxaspoly": "tab:orange", "fedcobra": "tab:red"}
labels = {"oracle": "Oracle", "polyonly": "Poly-only", "partial": "Partial", "masked": "Masked",
          "boxaspoly": "Box-as-Poly", "fedcobra": "FedCoBRA"}
for m in ORDER:
    cs = [np.array(R[f"main_s{s}_{m}"]["curve"]) for s in (0, 1) if f"main_s{s}_{m}" in R]
    if not cs:
        continue
    c = np.mean(np.stack([x[:, 3] for x in cs]), 0) * 100
    ax[0].plot(cs[0][:, 0], c, color=colors[m], lw=1.0, ls="--" if m == "oracle" else "-", label=labels[m])
ax[0].set_xlabel("round"); ax[0].set_ylabel("val cmIoU (%)"); ax[0].grid(alpha=0.3, lw=0.4)
ax[0].legend(ncol=2, frameon=False, handlelength=1.4, columnspacing=0.6, loc="lower right")
for s, st in ((0, "-"), (1, "--")):
    t = f"main_s{s}_fedcobra"
    if t in R and R[t]["rel_hist"]:
        rh = R[t]["rel_hist"]
        mr = [np.mean(list(d.values())) if d else np.nan for d in rh]
        ax[1].plot(np.arange(1, len(mr) + 1), mr, color="tab:red", ls=st, lw=1.0, label=f"seed {s}")
        macros[f"relfirst{s}"] = f"{np.nanmean(mr[:5]):.2f}"
        macros[f"rellast{s}"] = f"{np.nanmean(mr[-5:]):.2f}"
ax[1].set_xlabel("round"); ax[1].set_ylabel("mean box-client $r_k$"); ax[1].grid(alpha=0.3, lw=0.4)
ax[1].legend(frameon=False)
plt.tight_layout(pad=0.2, w_pad=0.6)
plt.savefig(f"{OUT}/figs/curves.pdf"); plt.close()

# ---------------------------------------------------------------- figure: rho sweep
fig, ax = plt.subplots(1, 1, figsize=(3.3, 1.45))
for m, col in (("masked", "tab:blue"), ("fedcobra", "tab:red")):
    xs, ys, yu = [], [], []
    for rho in (0.1, 0.3, 0.5):
        tag = f"main_s0_{m}" if rho == 0.3 else f"rho_s0_{rho}_{m}"
        if tag in R:
            xs.append(rho); ys.append(get(tag, "test", "cmiou") * 100); yu.append(get(tag, "unseen", "cmiou") * 100)
    if xs:
        ax.plot(xs, ys, "o-", color=col, ms=3, lw=1, label=f"{labels[m]} (seen)")
        ax.plot(xs, yu, "s--", color=col, ms=3, lw=1, label=f"{labels[m]} (unseen)")
if "main_s0_oracle" in R:
    ax.axhline(get("main_s0_oracle", "test", "cmiou") * 100, color="k", lw=0.8, ls=":", label="Oracle (seen)")
ax.set_xlabel("fraction of polygon clients $\\rho$"); ax.set_ylabel("test cmIoU (%)")
ax.set_xticks([0.1, 0.3, 0.5]); ax.grid(alpha=0.3, lw=0.4)
ax.legend(ncol=2, frameon=False, fontsize=5.5, handlelength=1.6, columnspacing=0.8)
plt.tight_layout(pad=0.2)
plt.savefig(f"{OUT}/figs/rho.pdf"); plt.close()

# ---------------------------------------------------------------- figure: data samples
from fedcobra.synpoly import PHI, CLASS_NAMES
from fedcobra.fl import get_data
clients = list(get_data(0, 300, 0.5)[0])
fig, ax = plt.subplots(1, 8, figsize=(7.1, 1.0))
for j in range(8):
    d = clients[j]; i = 3
    ax[j].imshow(d["x"][i], extent=(0, 1, 1, 0))
    c, r = d["c"][i], d["r"][i]
    px = np.r_[c[0] + r * np.cos(PHI), c[0] + r[0]]; py = np.r_[c[1] + r * np.sin(PHI), c[1]]
    ax[j].plot(px, py, "-", color="yellow", lw=0.8)
    b = d["box"][i]
    ax[j].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="cyan", lw=0.8, ls="--"))
    ax[j].set_title(f"client {j}: {CLASS_NAMES[int(d['y'][i])]}", fontsize=5.5, pad=1.5)
    ax[j].axis("off")
plt.tight_layout(pad=0.1, w_pad=0.2)
plt.savefig(f"{OUT}/figs/samples.pdf"); plt.close()

# ---------------------------------------------------------------- macros
with open(f"{OUT}/results_macros.tex", "w") as f:
    for k, v in macros.items():
        words = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}
        name = "R" + "".join(words.get(ch, ch) for ch in k if ch.isalnum())
        f.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")
print("results available:", len(R), sorted(R))
