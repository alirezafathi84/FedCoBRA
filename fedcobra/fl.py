"""Federated polygon/box instance-segmentation experiments (JAX, CPU).

Implements a compact polar-contour network (PolarMask-style ray regression on a
GroupNorm CNN), the weak box losses used on box-annotated clients, and the
aggregation rules: FedAvg, masked FedAvg, FedProx, FedAvgM, FedAdam, SCAFFOLD and
FedCoBRA's reliability-aware decoupled aggregation (RADA).
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import json, time, argparse
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from .synpoly import (IMG, G, N_RAYS, NUM_CLASSES, PHI, XX, YY, GX, GY, AFF_DELTA, make_federation)

F32 = jnp.float32
D = jnp.asarray(np.stack([np.cos(PHI), np.sin(PHI)], -1), F32)          # (N, 2) ray directions
GXj, GYj = jnp.asarray(GX, F32), jnp.asarray(GY, F32)
XXj, YYj = jnp.asarray(XX, F32), jnp.asarray(YY, F32)
CONTOUR = ("ct1_w", "ct1_b", "ct2_w", "ct2_b")
H_DESC = 8

# ----------------------------------------------------------------------------- model
# CoBRA-Net-tiny: GroupNorm CNN -> 8x8 feature map (receptive field > image).
# Detection head (per cell): object-centre logit, sub-cell offset, box size, class logits.
# Contour head (theta_C): MLP on the feature of the object-centre cell -> polar origin offset + N rays.
CH = [3, 12, 24, 24, 48, 48, 48]
STRIDE = [2, 2, 1, 2, 1, 1]
DIL = [1, 1, 1, 1, 2, 4]
S = IMG // 8                 # 8x8 output grid
DET_OUT = 1 + 2 + 2 + NUM_CLASSES


def init_params(seed):
    ks = jax.random.split(jax.random.PRNGKey(seed), 16)
    P = {}
    for i in range(6):
        std = np.sqrt(2.0 / (9 * CH[i]))
        P[f"conv{i}_w"] = jax.random.normal(ks[i], (3, 3, CH[i], CH[i + 1])) * std
        P[f"conv{i}_b"] = jnp.zeros(CH[i + 1])
        P[f"gn{i}_g"] = jnp.ones(CH[i + 1])
        P[f"gn{i}_b"] = jnp.zeros(CH[i + 1])

    def fc(k, a, b, s=1.0):
        return jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a) * s
    P["det_w"], P["det_b"] = fc(ks[8], CH[-1], DET_OUT, 0.1), jnp.zeros(DET_OUT)
    P["ct1_w"], P["ct1_b"] = fc(ks[9], CH[-1], 96), jnp.zeros(96)
    P["ct2_w"], P["ct2_b"] = fc(ks[10], 96, 2 + N_RAYS, 0.1), jnp.zeros(2 + N_RAYS)
    return jax.tree_util.tree_map(lambda a: a.astype(F32), P)


def n_params(P, keys=None):
    return int(sum(np.prod(v.shape) for k, v in P.items() if keys is None or k in keys))


def gn(x, g, b, groups=4, eps=1e-5):
    B, H, W, C = x.shape
    xg = x.reshape(B, H, W, groups, C // groups)
    m = xg.mean(axis=(1, 2, 4), keepdims=True)
    v = xg.var(axis=(1, 2, 4), keepdims=True)
    return ((xg - m) * jax.lax.rsqrt(v + eps)).reshape(B, H, W, C) * g + b


def backbone(P, x):
    h = x
    for i in range(6):
        h = jax.lax.conv_general_dilated(h, P[f"conv{i}_w"], (STRIDE[i], STRIDE[i]), "SAME",
                                         rhs_dilation=(DIL[i], DIL[i]),
                                         dimension_numbers=("NHWC", "HWIO", "NHWC")) + P[f"conv{i}_b"]
        h = jax.nn.relu(gn(h, P[f"gn{i}_g"], P[f"gn{i}_b"]))
    return h                                     # (B, 8, 8, C)


def forward(P, x, cell=None):
    """If cell is None the predicted (argmax) centre cell is used (inference)."""
    h = backbone(P, x)
    B = h.shape[0]
    feat = h.reshape(B, S * S, -1)
    det = feat @ P["det_w"] + P["det_b"]       # (B, 64, DET_OUT)
    loc = det[..., 0]
    if cell is None:
        cell = jnp.argmax(loc, -1)
    v = jnp.take_along_axis(det, cell[:, None, None], axis=1)[:, 0]
    f = jnp.take_along_axis(feat, cell[:, None, None], axis=1)[:, 0]
    off = 0.5 * jnp.tanh(v[:, 1:3])
    wh = jax.nn.sigmoid(v[:, 3:5]) * 0.95
    logits = v[:, 5:]
    gx = (cell % S).astype(F32); gy = (cell // S).astype(F32)
    cx = (gx + 0.5 + off[:, 0]) / S; cy = (gy + 0.5 + off[:, 1]) / S
    box = jnp.stack([cx - wh[:, 0] / 2, cy - wh[:, 1] / 2, cx + wh[:, 0] / 2, cy + wh[:, 1] / 2], -1)
    g = jax.nn.relu(f @ P["ct1_w"] + P["ct1_b"])
    co = g @ P["ct2_w"] + P["ct2_b"]
    p0 = jnp.stack([cx, cy], -1) + 0.15 * jnp.tanh(co[:, :2])
    r = 0.2 * jax.nn.softplus(co[:, 2:]) + 0.005
    return loc, logits, box, p0, r


def gt_cell(box):
    cx = (box[:, 0] + box[:, 2]) / 2; cy = (box[:, 1] + box[:, 3]) / 2
    return (jnp.clip(jnp.floor(cy * S), 0, S - 1) * S + jnp.clip(jnp.floor(cx * S), 0, S - 1)).astype(jnp.int32)

# ----------------------------------------------------------------------------- geometry

def interp_r(r, ang):
    pos = ang * (N_RAYS / (2 * np.pi))
    i0f = jnp.floor(pos)
    f = pos - i0f
    i0 = i0f.astype(jnp.int32) % N_RAYS
    i1 = (i0 + 1) % N_RAYS
    B = r.shape[0]
    r0 = jnp.take_along_axis(r, i0.reshape(B, -1), axis=1).reshape(ang.shape)
    r1 = jnp.take_along_axis(r, i1.reshape(B, -1), axis=1).reshape(ang.shape)
    return r0 * (1 - f) + r1 * f


def polar_field(p0, r, gx, gy):
    """Signed radial margin: > 0 inside the predicted polygon."""
    dx = gx[None] - p0[:, 0, None, None]
    dy = gy[None] - p0[:, 1, None, None]
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-12)
    ang = jnp.mod(jnp.arctan2(dy, dx), 2 * np.pi)
    return interp_r(r, ang) - dist


def verts(p0, r):
    return p0[:, None, :] + r[..., None] * D[None]


def fdesc(r):
    """Rotation/scale/translation-invariant centroid-distance Fourier descriptor."""
    s = r / jnp.mean(r, -1, keepdims=True)
    F = jnp.fft.rfft(s, axis=-1)[:, 1:H_DESC + 1]
    return jnp.sqrt(F.real ** 2 + F.imag ** 2 + 1e-10) / N_RAYS


def fdesc_np(r):
    s = r / r.mean(-1, keepdims=True)
    return np.abs(np.fft.rfft(s, axis=-1)[:, 1:H_DESC + 1]) / N_RAYS


def box_iou(a, b):
    ix = jnp.clip(jnp.minimum(a[:, 2], b[:, 2]) - jnp.maximum(a[:, 0], b[:, 0]), 0)
    iy = jnp.clip(jnp.minimum(a[:, 3], b[:, 3]) - jnp.maximum(a[:, 1], b[:, 1]), 0)
    inter = ix * iy
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / jnp.maximum(ua, 1e-8)

# ----------------------------------------------------------------------------- losses

def smax(x, tau):
    return tau * jax.nn.logsumexp(x / tau, axis=-1)


def l_tight(V, box, tau=0.01):
    vx, vy = V[..., 0], V[..., 1]
    return (jnp.abs(-smax(-vx, tau) - box[:, 0]) + jnp.abs(-smax(-vy, tau) - box[:, 1]) +
            jnp.abs(smax(vx, tau) - box[:, 2]) + jnp.abs(smax(vy, tau) - box[:, 3])) / 4


def l_in(V, box):
    vx, vy = V[..., 0], V[..., 1]
    v = (jax.nn.relu(box[:, 0, None] - vx) + jax.nn.relu(vx - box[:, 2, None]) +
         jax.nn.relu(box[:, 1, None] - vy) + jax.nn.relu(vy - box[:, 3, None]))
    return v.mean(-1)


def l_aff(p0, r, eh, ev, sigma=0.5 / G):
    m = jax.nn.sigmoid(polar_field(p0, r, GXj, GYj) / sigma)
    ph = m[:, :, 1:] * m[:, :, :-1] + (1 - m[:, :, 1:]) * (1 - m[:, :, :-1])
    pv = m[:, 1:, :] * m[:, :-1, :] + (1 - m[:, 1:, :]) * (1 - m[:, :-1, :])
    num = (-jnp.log(ph + 1e-6) * eh).sum((1, 2)) + (-jnp.log(pv + 1e-6) * ev).sum((1, 2))
    return num / (eh.sum((1, 2)) + ev.sum((1, 2)) + 1.0)


def l_shape(r, y, mu, var, has):
    z = fdesc(r)
    d2 = ((z - mu[y]) ** 2 / (var[y] + 1e-5)).mean(-1)
    return d2 / (d2 + 1.0) * has[y]


def l_distill(V, Vt, box, gamma):
    tx1, tx2 = Vt[..., 0].min(-1), Vt[..., 0].max(-1)
    ty1, ty2 = Vt[..., 1].min(-1), Vt[..., 1].max(-1)
    q = box_iou(jnp.stack([tx1, ty1, tx2, ty2], -1), box)
    sx = (box[:, 2] - box[:, 0]) / jnp.maximum(tx2 - tx1, 1e-4)
    sy = (box[:, 3] - box[:, 1]) / jnp.maximum(ty2 - ty1, 1e-4)
    Vr = jnp.stack([box[:, 0, None] + (Vt[..., 0] - tx1[:, None]) * sx[:, None],
                    box[:, 1, None] + (Vt[..., 1] - ty1[:, None]) * sy[:, None]], -1)
    Vr = jax.lax.stop_gradient(Vr)
    w = jax.lax.stop_gradient(q * (q >= gamma))
    return w * jnp.abs(V - Vr).sum(-1).mean(-1)


def box_as_poly(box):
    cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    hw, hh = (box[:, 2] - box[:, 0]) / 2, (box[:, 3] - box[:, 1]) / 2
    r = jnp.minimum(hw[:, None] / jnp.maximum(jnp.abs(D[:, 0])[None], 1e-6),
                    hh[:, None] / jnp.maximum(jnp.abs(D[:, 1])[None], 1e-6))
    return jnp.stack([cx, cy], -1), r


def make_loss(mode, flags, lam):
    tight, aff, shape, distill = flags
    lam_aff, lam_shape, lam_d, gamma = lam

    def loss(P, x, y, box_t, c_t, r_t, eh, ev, Pt, proto, Pg, prox):
        cell = gt_cell(box_t)
        loc, logits, box, p0, r = forward(P, x, cell)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], axis=1)[:, 0]
        lce = -jnp.take_along_axis(jax.nn.log_softmax(loc), cell[:, None], axis=1)[:, 0]
        L = ce + lce + 10.0 * jnp.abs(box - box_t).mean(-1)
        if mode == "poly":
            L = L + 10.0 * (jnp.abs(p0 - c_t).mean(-1) + jnp.abs(r - r_t).mean(-1))
        elif mode == "boxaspoly":
            c2, r2 = box_as_poly(box_t)
            L = L + 10.0 * (jnp.abs(p0 - c2).mean(-1) + jnp.abs(r - r2).mean(-1))
        elif mode == "weak":
            V = verts(p0, r)
            if tight:
                L = L + 10.0 * (l_tight(V, box_t) + l_in(V, box_t))
            if aff:
                L = L + lam_aff * l_aff(p0, r, eh.astype(F32), ev.astype(F32))
            if shape:
                L = L + lam_shape * l_shape(r, y, *proto)
            if distill:
                _, _, _, p0t, rt = forward(Pt, x, cell)
                L = L + 10.0 * lam_d * l_distill(V, verts(p0t, rt), box_t, gamma)
        L = L.mean()
        sq = sum(jnp.sum((P[k] - Pg[k]) ** 2) for k in P)
        return L + 0.5 * prox * sq
    return loss


@partial(jax.jit, static_argnames=("mode", "flags", "lam", "opt"))
def train_step(P, m, v, t, x, y, box_t, c_t, r_t, eh, ev, Pt, proto, lr, prox, c_glob, c_loc,
               mode, flags, lam, opt):
    g = jax.grad(make_loss(mode, flags, lam))(P, x, y, box_t, c_t, r_t, eh, ev, Pt, proto, Pt, prox)
    if opt == "adam":
        t = t + 1
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        P = jax.tree_util.tree_map(
            lambda p, a, b: p - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8), P, m, v)
    elif opt == "sgd":
        P = jax.tree_util.tree_map(lambda p, a: p - lr * a, P, g)
    elif opt == "scaffold":
        P = jax.tree_util.tree_map(lambda p, a, cg, cl: p - lr * (a - cl + cg), P, g, c_glob, c_loc)
    return P, m, v, t


RAY = np.arange(N_RAYS)


def augment_batch(d, ib, rng, geo=True, color=True, masks=None):
    """D4 geometric + colour augmentation with exact label transforms; recomputes the affinity graph."""
    x = d["x"][ib].astype(np.float32) / 255.0
    box = d["box"][ib].copy(); c = d["c"][ib].copy(); r = d["r"][ib].copy()
    mk = None if masks is None else masks[ib].copy()
    B = len(ib)
    for i in range(B):
        if geo:
            if rng.random() < 0.5:
                x[i] = x[i].transpose(1, 0, 2); box[i] = box[i][[1, 0, 3, 2]]; c[i] = c[i][::-1]
                r[i] = r[i][(8 - RAY) % N_RAYS]
                if mk is not None: mk[i] = mk[i].T
            if rng.random() < 0.5:
                x[i] = x[i][:, ::-1]; box[i] = [1 - box[i][2], box[i][1], 1 - box[i][0], box[i][3]]
                c[i][0] = 1 - c[i][0]; r[i] = r[i][(16 - RAY) % N_RAYS]
                if mk is not None: mk[i] = mk[i][:, ::-1]
            if rng.random() < 0.5:
                x[i] = x[i][::-1, :]; box[i] = [box[i][0], 1 - box[i][3], box[i][2], 1 - box[i][1]]
                c[i][1] = 1 - c[i][1]; r[i] = r[i][(-RAY) % N_RAYS]
                if mk is not None: mk[i] = mk[i][::-1, :]
        if color:
            x[i] = x[i][..., rng.permutation(3)]
            if rng.random() < 0.15:
                x[i] = np.repeat(x[i].mean(-1, keepdims=True), 3, -1)
            mu = x[i].mean()
            x[i] = np.clip((x[i] - mu) * rng.uniform(0.85, 1.15) + mu + rng.uniform(-0.08, 0.08), 0, 1)
    sm = x.reshape(B, G, IMG // G, G, IMG // G, 3).mean((2, 4))
    dh = np.linalg.norm(sm[:, :, 1:] - sm[:, :, :-1], axis=-1)
    dv = np.linalg.norm(sm[:, 1:, :] - sm[:, :-1, :], axis=-1)
    m_ = 1.0 / G
    inb = ((GX[None] >= box[:, 0, None, None] - m_) & (GX[None] <= box[:, 2, None, None] + m_) &
           (GY[None] >= box[:, 1, None, None] - m_) & (GY[None] <= box[:, 3, None, None] + m_))
    eh = (dh < AFF_DELTA) & inb[:, :, 1:] & inb[:, :, :-1]
    ev = (dv < AFF_DELTA) & inb[:, 1:, :] & inb[:, :-1, :]
    out = dict(x=x, y=d["y"][ib], box=box.astype(np.float32), c=c.astype(np.float32), r=r.astype(np.float32),
               eh=eh, ev=ev)
    if mk is not None:
        out["mask"] = mk
    return out


def local_train(P, d, idx, Pt, proto, lr, prox, c_glob, c_loc, mode, flags, lam, opt, rng, aug=True):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, P)
    m, v, t = zeros, zeros, jnp.asarray(0.0, F32)
    for ib in idx:
        b = augment_batch(d, ib, rng, geo=aug, color=aug)
        P, m, v, t = train_step(P, m, v, t, b["x"], b["y"], b["box"], b["c"], b["r"], b["eh"], b["ev"],
                                Pt, proto, lr, prox, c_glob, c_loc, mode=mode, flags=flags, lam=lam, opt=opt)
    return P

# ----------------------------------------------------------------------------- evaluation

@jax.jit
def _eval_batch(P, x, mask, y, box_t):
    xf = x.astype(F32) / 255.0
    loc, logits, box, p0, r = forward(P, xf)
    pm = polar_field(p0, r, XXj, YYj) > 0
    iou = (pm & mask).sum((1, 2)) / jnp.maximum((pm | mask).sum((1, 2)), 1)
    correct = jnp.argmax(logits, -1) == y
    _, _, _, p0g, rg = forward(P, xf, gt_cell(box_t))          # contour quality given the true centre cell
    pg = polar_field(p0g, rg, XXj, YYj) > 0
    ciou = (pg & mask).sum((1, 2)) / jnp.maximum((pg | mask).sum((1, 2)), 1)
    return iou, correct, box_iou(box, box_t), ciou


def evaluate(P, split, bs=300):
    ious, cor, bious, cious = [], [], [], []
    for s in range(0, len(split["y"]), bs):
        a, b, c, e = _eval_batch(P, split["x"][s:s + bs], split["mask"][s:s + bs], split["y"][s:s + bs],
                                 split["box"][s:s + bs])
        ious.append(np.asarray(a)); cor.append(np.asarray(b)); bious.append(np.asarray(c)); cious.append(np.asarray(e))
    iou, cor, biou, ciou = (np.concatenate(z) for z in (ious, cor, bious, cious))
    return dict(miou=float(iou.mean()), r50=float((iou >= 0.5).mean()), r75=float((iou >= 0.75).mean()),
                cmiou=float(ciou.mean()), cr75=float((ciou >= 0.75).mean()),
                cls=float(cor.mean()), box_iou=float(biou.mean()),
                per_class_cmiou=[float(ciou[split["y"] == c].mean()) for c in range(NUM_CLASSES)],
                per_class_miou=[float(iou[split["y"] == c].mean()) for c in range(NUM_CLASSES)])

# ----------------------------------------------------------------------------- federated shape prototypes

def shape_stats(Rgt, Y, clipC=0.6, eps=None, delta=1e-5, rng=None):
    z = fdesc_np(Rgt)
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    z = z * np.minimum(1.0, clipC / np.maximum(nrm, 1e-12))
    S = np.zeros((NUM_CLASSES, H_DESC)); Q = np.zeros((NUM_CLASSES, H_DESC)); n = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        S[c] = z[Y == c].sum(0); Q[c] = (z[Y == c] ** 2).sum(0); n[c] = (Y == c).sum()
    if eps is not None:
        sens = np.sqrt(clipC ** 2 + clipC ** 4 + 1.0)
        sigma = sens * np.sqrt(2 * np.log(1.25 / delta)) / eps
        S = S + rng.normal(0, sigma, S.shape); Q = Q + rng.normal(0, sigma, Q.shape)
        n = n + rng.normal(0, sigma, n.shape)
    return S, Q, n


def prototypes(stats):
    S = sum(s[0] for s in stats); Q = sum(s[1] for s in stats); n = sum(s[2] for s in stats)
    nn_ = np.maximum(n, 1.0)
    mu = S / nn_[:, None]
    var = np.maximum(Q / nn_[:, None] - mu ** 2, 1e-4)
    has = (n >= 5).astype(np.float32)
    return (jnp.asarray(mu, F32), jnp.asarray(var, F32), jnp.asarray(has, F32))

# ----------------------------------------------------------------------------- aggregation helpers

def tree_wavg(Ps, w):
    w = np.asarray(w, np.float64); w = w / w.sum()
    return jax.tree_util.tree_map(lambda *a: sum(wi * ai for wi, ai in zip(w, a)), *Ps)


def flat(P, keys):
    return jnp.concatenate([P[k].ravel() for k in keys])


def unflat(vec, like, keys):
    out, o = {}, 0
    for k in keys:
        sz = int(np.prod(like[k].shape)); out[k] = vec[o:o + sz].reshape(like[k].shape); o += sz
    return out


def rada(Pg, Ps, n, types, beta):
    """Reliability-aware decoupled aggregation. Returns new params and box-client reliabilities."""
    shared = [k for k in Pg if k not in CONTOUR]
    newP = dict(tree_wavg([{k: P[k] for k in shared} for P in Ps], n))
    g = flat(Pg, CONTOUR)
    deltas = [flat(P, CONTOUR) - g for P in Ps]
    pidx = [i for i, t in enumerate(types) if t == "poly"]
    rel = {}
    if not pidx:  # no anchor available: plain FedAvg
        newP.update({k: v for k, v in tree_wavg([{k: P[k] for k in CONTOUR} for P in Ps], n).items()})
        return newP, rel
    dP = sum(n[i] * deltas[i] for i in pidx) / sum(n[i] for i in pidx)
    nP = jnp.linalg.norm(dP) + 1e-12
    agg, wsum = jnp.zeros_like(g), 0.0
    for i, d in enumerate(deltas):
        if types[i] == "poly":
            agg = agg + n[i] * d; wsum += n[i]
        else:
            nd = jnp.linalg.norm(d) + 1e-12
            cos = float(jnp.dot(d, dP) / (nd * nP))
            r = max(cos, 0.0); rel[i] = r
            if r > 0:
                agg = agg + beta * n[i] * r * d * jnp.minimum(1.0, nP / nd); wsum += beta * n[i] * r
    newP.update(unflat(g + agg / wsum, Pg, CONTOUR))
    return newP, rel

# ----------------------------------------------------------------------------- experiment driver

DATA_CACHE = {}


def get_data(seed, n_per_client, alpha):
    key = (seed, n_per_client, alpha)
    if key not in DATA_CACHE:
        path = os.path.join(os.environ.get("FEDCOBRA_CACHE", "cache"), f"fed_s{seed}_n{n_per_client}_a{alpha}.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            clients = list(z["clients"]); val, test, unseen = z["val"].item(), z["test"].item(), z["unseen"].item()
        else:
            clients, val, test, unseen, _ = make_federation(seed, n_per_client=n_per_client, alpha=alpha)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, clients=np.array(clients, dtype=object), val=val, test=test, unseen=unseen)
        DATA_CACHE[key] = (clients, val, test, unseen)
    return DATA_CACHE[key]


def run(cfg):
    t0 = time.time()
    K = 10
    clients, val, test, unseen = get_data(cfg["seed"], cfg["n_pool"], cfg["alpha"])
    perm = np.random.default_rng([cfg["seed"], 99]).permutation(K)
    n_poly = int(round(cfg["rho"] * K))
    is_poly = np.zeros(K, bool); is_poly[perm[:n_poly]] = True
    method = cfg["method"]
    if method == "oracle":
        is_poly[:] = True
    part = [k for k in range(K) if not (method == "polyonly" and not is_poly[k])]
    sizes = [cfg["n_poly_img"] if is_poly[k] else cfg["n_box_img"] for k in range(K)]
    data = []
    for k in range(K):
        d = clients[k]; m = sizes[k]
        data.append({kk: np.ascontiguousarray(v[:m]) for kk, v in d.items() if kk != "mask"})
    box_mode = {"partial": "partial", "masked": "partial", "boxaspoly": "boxaspoly",
                "fedcobra": "weak", "polyonly": "partial", "oracle": "poly"}.get(method, "weak")
    flags = (cfg["tight"], cfg["aff"], cfg["shape"], cfg["distill"])
    lam = (cfg["lam_aff"], cfg["lam_shape"], cfg["lam_d"], cfg["gamma"])
    # federated shape prototypes (computed once, before training; optional DP)
    dp_rng = np.random.default_rng([cfg["seed"], 5])
    stats = [shape_stats(np.asarray(clients[k]["r"][:sizes[k]]), np.asarray(clients[k]["y"][:sizes[k]]),
                         eps=cfg["dp_eps"], rng=dp_rng) for k in range(K) if is_poly[k]]
    proto = prototypes(stats) if stats else (jnp.zeros((NUM_CLASSES, H_DESC)), jnp.ones((NUM_CLASSES, H_DESC)),
                                            jnp.zeros(NUM_CLASSES))
    P = init_params(cfg["seed"])
    zeros = jax.tree_util.tree_map(jnp.zeros_like, P)
    c_glob, c_loc = zeros, [zeros for _ in range(K)]
    mom = zeros; adam_m, adam_v = zeros, zeros
    rng = np.random.default_rng([cfg["seed"], 3])
    curve, rel_hist = [], []
    opt = cfg["local_opt"]
    for t in range(cfg["rounds"]):
        Ps, ns, types = [], [], []
        for k in part:
            mode = "poly" if is_poly[k] else box_mode
            n_k = sizes[k]
            idx = rng.integers(0, n_k, size=(cfg["local_steps"], cfg["batch"]))
            Pk = local_train(P, data[k], idx, P, proto, cfg["lr"], cfg["prox"], c_glob, c_loc[k], mode=mode,
                             flags=flags if mode == "weak" else (False,) * 4,
                             lam=lam if mode == "weak" else (0.0,) * 4, opt=opt, rng=rng,
                             aug=cfg["aug"])
            if opt == "scaffold":
                ck = jax.tree_util.tree_map(lambda cl, cg, pg, pk: cl - cg + (pg - pk) / (cfg["local_steps"] * cfg["lr"]),
                                            c_loc[k], c_glob, P, Pk)
                c_loc[k] = ck
            Ps.append(Pk); ns.append(float(n_k)); types.append("poly" if is_poly[k] else "box")
        agg = cfg["agg"]
        if agg == "rada":
            newP, rel = rada(P, Ps, ns, types, cfg["beta"]); rel_hist.append({int(part[i]): v for i, v in rel.items()})
        elif agg == "masked":
            newP = dict(tree_wavg([{k: Q[k] for k in P if k not in CONTOUR} for Q in Ps], ns))
            pi = [i for i, tp in enumerate(types) if tp == "poly"]
            newP.update(tree_wavg([{k: Ps[i][k] for k in CONTOUR} for i in pi], [ns[i] for i in pi]))
        else:
            newP = tree_wavg(Ps, ns)
        if cfg["server_opt"] == "avgm":
            delta = jax.tree_util.tree_map(lambda a, b: a - b, newP, P)
            mom = jax.tree_util.tree_map(lambda m, d: cfg["server_mom"] * m + d, mom, delta)
            P = jax.tree_util.tree_map(lambda p, m: p + cfg["server_lr"] * m, P, mom)
        elif cfg["server_opt"] == "adam":
            delta = jax.tree_util.tree_map(lambda a, b: a - b, newP, P)
            adam_m = jax.tree_util.tree_map(lambda m, d: 0.9 * m + 0.1 * d, adam_m, delta)
            adam_v = jax.tree_util.tree_map(lambda v, d: 0.99 * v + 0.01 * d * d, adam_v, delta)
            P = jax.tree_util.tree_map(lambda p, m, v: p + cfg["server_lr"] * m / (jnp.sqrt(v) + 1e-3), P, adam_m, adam_v)
        else:
            P = newP
        if opt == "scaffold":
            # c <- c + (1/N) sum_i (c_i^+ - c_i); full participation => c = mean(c_i)
            c_glob = tree_wavg([c_loc[k] for k in part], [1.0] * len(part))
        if (t + 1) % cfg["eval_every"] == 0 or t == cfg["rounds"] - 1:
            ev = evaluate(P, val); curve.append((t + 1, ev["miou"], ev["r75"], ev["cmiou"]))
    res = dict(cfg=cfg, curve=curve, rel_hist=rel_hist, val=evaluate(P, val), test=evaluate(P, test),
               unseen=evaluate(P, unseen), n_params=n_params(P), n_params_contour=n_params(P, CONTOUR),
               poly_clients=[int(k) for k in np.where(is_poly)[0]], time=time.time() - t0)
    return res


DEFAULT = dict(method="fedcobra", seed=0, rho=0.3, alpha=0.5, n_pool=300, n_poly_img=300, n_box_img=300,
               rounds=40, local_steps=12, batch=16, lr=2e-3, local_opt="adam", prox=0.0,
               agg="rada", beta=1.0, server_opt="none", server_lr=1.0, server_mom=0.9,
               tight=True, aff=True, shape=True, distill=True, lam_aff=1.0, lam_shape=0.5, lam_d=1.0,
               gamma=0.5, dp_eps=None, eval_every=5, aug=True, tag="")


def method_cfg(method, **kw):
    c = dict(DEFAULT); c["method"] = method
    if method in ("oracle", "polyonly", "partial", "boxaspoly"):
        c["agg"] = "fedavg"
    if method == "masked":
        c["agg"] = "masked"
    c.update(kw)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    os.makedirs(a.out, exist_ok=True)
    for cfg in plan:
        fn = os.path.join(a.out, cfg["tag"] + ".json")
        if os.path.exists(fn):
            continue
        res = run(cfg)
        json.dump(res, open(fn, "w"))
        print(cfg["tag"], "test mIoU %.4f cmIoU %.4f r75 %.4f box %.3f cls %.3f | unseen mIoU %.4f cmIoU %.4f (%.0fs)" %
              (res["test"]["miou"], res["test"]["cmiou"], res["test"]["r75"], res["test"]["box_iou"], res["test"]["cls"],
               res["unseen"]["miou"], res["unseen"]["cmiou"], res["time"]), flush=True)
