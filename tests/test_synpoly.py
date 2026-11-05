import numpy as np
from fedcobra.synpoly import sample_domain, make_split, XX, YY, N_RAYS, NUM_CLASSES, IMG


def raster(c, r):
    dx, dy = XX - c[0], YY - c[1]
    d = np.sqrt(dx ** 2 + dy ** 2)
    a = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    pos = a / (2 * np.pi) * N_RAYS
    i0 = np.floor(pos).astype(int) % N_RAYS
    f = pos - np.floor(pos)
    return d <= r[i0] * (1 - f) + r[(i0 + 1) % N_RAYS] * f


def _split(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return make_split(rng, sample_domain(rng), np.ones(NUM_CLASSES) / NUM_CLASSES, n)


def test_polygon_label_matches_mask():
    s = _split()
    ious = [(raster(s["c"][i], s["r"][i]) & s["mask"][i]).sum() / (raster(s["c"][i], s["r"][i]) | s["mask"][i]).sum()
            for i in range(len(s["y"]))]
    assert np.mean(ious) > 0.95


def test_box_is_tight():
    s = _split()
    for i in range(len(s["y"])):
        ys, xs = np.where(s["mask"][i])
        ext = np.array([xs.min(), ys.min(), xs.max(), ys.max()]) + 0.5
        assert np.abs(ext / IMG - s["box"][i]).max() * IMG < 2.0
