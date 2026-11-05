"""SynPoly-FL: a small synthetic federated benchmark for polygon vs. box supervision.

Each image contains one star-convex object from one of four shape families.
Every client has its own visual domain (colours, stripes, noise, illumination,
distractors) and its own Dirichlet class prior, so the federation is non-IID in
both feature space and label space. Ground truth is stored both as a polygon
(radial distances on N_RAYS fixed rays) and as a tight bounding box, so the same
images can be given to a client as polygon labels or as box labels.
"""
import numpy as np

IMG = 64            # image side (pixels)
G = 32              # grid used by the weak (affinity) loss
N_RAYS = 32         # polygon vertices (fixed polar rays)
NUM_CLASSES = 4
CLASS_NAMES = ["ellipse", "n-gon", "star", "blob"]
PHI = 2 * np.pi * np.arange(N_RAYS) / N_RAYS
AFF_DELTA = 0.11    # colour-distance threshold for the pairwise affinity graph

_c = (np.arange(IMG) + 0.5) / IMG
YY, XX = np.meshgrid(_c, _c, indexing="ij")
_g = (np.arange(G) + 0.5) / G
GY, GX = np.meshgrid(_g, _g, indexing="ij")


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def radial(cls, p, phi):
    """Distance from the shape centre to its boundary along absolute angle phi."""
    t = np.mod(phi - p["rot"], 2 * np.pi)
    R = p["R"]
    if cls == 0:  # ellipse
        a, b = R, R * p["ar"]
        return a * b / np.sqrt((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2)
    if cls == 1:  # regular k-gon
        k = p["k"]
        seg = 2 * np.pi / k
        tt = np.mod(t, seg) - seg / 2
        return R * np.cos(np.pi / k) / np.cos(tt)
    if cls == 2:  # five-point star (10-vertex polygon)
        K, rin = 10, R * p["inner"]
        seg = 2 * np.pi / K
        j = np.floor(t / seg).astype(int) % K
        rA = np.where(j % 2 == 0, R, rin)
        rB = np.where(j % 2 == 0, rin, R)
        aA = j * seg
        aB = aA + seg
        A = np.stack([rA * np.cos(aA), rA * np.sin(aA)], -1)
        B = np.stack([rB * np.cos(aB), rB * np.sin(aB)], -1)
        d = np.stack([np.cos(t), np.sin(t)], -1)
        BA = B - A
        return _cross(A, BA) / _cross(d, BA)
    # blob: low-order Fourier perturbation of a circle
    r = np.ones_like(t)
    for h, (amp, ph) in zip((2, 3, 4), p["harm"]):
        r = r + amp * np.cos(h * t + ph)
    return R * r


def sample_params(rng, cls):
    p = {"R": rng.uniform(0.13, 0.27), "rot": rng.uniform(0, 2 * np.pi)}
    if cls == 0:
        p["ar"] = rng.uniform(0.35, 0.9)
    elif cls == 1:
        p["k"] = int(rng.choice([3, 4, 5]))
    elif cls == 2:
        p["inner"] = rng.uniform(0.4, 0.6)
    else:
        p["harm"] = [(rng.uniform(0.05, 0.18), rng.uniform(0, 2 * np.pi)) for _ in range(3)]
    return p


def sample_domain(rng):
    """A client's visual domain (feature skew)."""
    bg = rng.uniform(0.15, 0.85, 3)
    contrast = rng.uniform(0.22, 0.55)
    obj = None
    for _ in range(200):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        cand = bg + contrast * d
        if np.all(cand >= 0.02) and np.all(cand <= 0.98):
            obj = cand
            break
    if obj is None:
        obj = np.clip(bg + contrast * d, 0.02, 0.98)
    return dict(bg=bg, obj=obj, contrast=float(np.linalg.norm(obj - bg)),
                stripe_amp=rng.uniform(0.0, 0.12), stripe_freq=rng.uniform(2, 7),
                stripe_theta=rng.uniform(0, np.pi), noise=rng.uniform(0.02, 0.07),
                grad_amp=rng.uniform(0.0, 0.15), obj_tex=rng.uniform(0.0, 0.08),
                n_distr=int(rng.integers(0, 3)))


def render(rng, dom, cls, p, c):
    th = dom["stripe_theta"]
    proj = XX * np.cos(th) + YY * np.sin(th)
    stripes = dom["stripe_amp"] * np.sin(2 * np.pi * dom["stripe_freq"] * proj + rng.uniform(0, 2 * np.pi))
    bgcol = np.clip(dom["bg"] + rng.normal(0, 0.03, 3), 0, 1)
    img = bgcol[None, None, :] + stripes[..., None]
    for _ in range(dom["n_distr"]):
        dc = rng.uniform(0.05, 0.95, 2)
        dr = rng.uniform(0.025, 0.05)
        m = (XX - dc[0]) ** 2 + (YY - dc[1]) ** 2 <= dr ** 2
        img[m] = rng.uniform(0, 1, 3)
    dx, dy = XX - c[0], YY - c[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)
    mask = dist <= radial(cls, p, ang)
    objcol = np.clip(dom["obj"] + rng.normal(0, 0.03, 3), 0, 1)
    tex = dom["obj_tex"] * np.sin(2 * np.pi * 6 * (XX + YY) + rng.uniform(0, 2 * np.pi))
    img[mask] = objcol[None, :] + tex[mask][:, None]
    gd = rng.uniform(0, 2 * np.pi)
    img = img + (dom["grad_amp"] * ((XX - 0.5) * np.cos(gd) + (YY - 0.5) * np.sin(gd)))[..., None]
    img = img + rng.normal(0, dom["noise"], img.shape)
    return np.clip(img, 0, 1), mask


def affinity_edges(img, box):
    """BoxInst-style colour-affinity graph on a G x G grid, restricted to the box."""
    s = img.reshape(G, IMG // G, G, IMG // G, 3).mean((1, 3))
    dh = np.linalg.norm(s[:, 1:] - s[:, :-1], axis=-1)
    dv = np.linalg.norm(s[1:, :] - s[:-1, :], axis=-1)
    m = 1.0 / G
    inb = (GX >= box[0] - m) & (GX <= box[2] + m) & (GY >= box[1] - m) & (GY <= box[3] + m)
    eh = (dh < AFF_DELTA) & inb[:, 1:] & inb[:, :-1]
    ev = (dv < AFF_DELTA) & inb[1:, :] & inb[:-1, :]
    return eh, ev


def sample_example(rng, dom, cls):
    p = sample_params(rng, cls)
    phis = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    rd = radial(cls, p, phis)
    xs, ys = rd * np.cos(phis), rd * np.sin(phis)
    cx = rng.uniform(0.03 - xs.min(), 0.97 - xs.max())
    cy = rng.uniform(0.03 - ys.min(), 0.97 - ys.max())
    box = np.array([cx + xs.min(), cy + ys.min(), cx + xs.max(), cy + ys.max()])
    img, mask = render(rng, dom, cls, p, (cx, cy))
    rays = radial(cls, p, PHI)
    eh, ev = affinity_edges(img, box)
    return img, mask, box, np.array([cx, cy]), rays, eh, ev


def make_split(rng, dom, probs, n):
    out = {k: [] for k in ["x", "mask", "box", "c", "r", "y", "eh", "ev"]}
    ys = rng.choice(NUM_CLASSES, size=n, p=probs)
    for y in ys:
        img, mask, box, c, r, eh, ev = sample_example(rng, dom, int(y))
        out["x"].append((img * 255).round().astype(np.uint8))
        out["mask"].append(mask)
        out["box"].append(box)
        out["c"].append(c)
        out["r"].append(r)
        out["y"].append(y)
        out["eh"].append(eh)
        out["ev"].append(ev)
    return {k: np.stack(v).astype(np.float32 if k in ("box", "c", "r") else
                                  (np.int32 if k == "y" else v[0].dtype)) for k, v in out.items()}


def concat(splits):
    return {k: np.concatenate([s[k] for s in splits]) for k in splits[0]}


def make_federation(seed, K=10, n_per_client=240, alpha=0.5, n_val=30, n_test=60, n_unseen=5):
    """Clients get a prefix of a client-specific stream so that budgets are nested."""
    root = np.random.default_rng(seed)
    domains = [sample_domain(root) for _ in range(K)]
    priors = [root.dirichlet(alpha * np.ones(NUM_CLASSES)) for _ in range(K)]
    unseen = [sample_domain(root) for _ in range(n_unseen)]
    uni = np.ones(NUM_CLASSES) / NUM_CLASSES
    clients = []
    for k in range(K):
        rng = np.random.default_rng([seed, 1000 + k])
        clients.append(make_split(rng, domains[k], priors[k], n_per_client))
    rng = np.random.default_rng([seed, 7])
    val = concat([make_split(rng, d, uni, n_val) for d in domains])
    test = concat([make_split(rng, d, uni, n_test) for d in domains])
    test_unseen = concat([make_split(rng, d, uni, n_test) for d in unseen])
    meta = dict(domains=domains, priors=[p.tolist() for p in priors], unseen=unseen)
    return clients, val, test, test_unseen, meta
