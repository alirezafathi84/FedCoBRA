import numpy as np
import jax.numpy as jnp
from fedcobra.fl import augment_batch, init_params, rada, flat, CONTOUR
from tests.test_synpoly import raster, _split


def test_augmentation_transforms_labels_exactly():
    s = _split(n=32, seed=1)
    b = augment_batch(s, np.arange(32), np.random.default_rng(0), masks=s["mask"])
    ious = [(raster(b["c"][i], b["r"][i]) & b["mask"][i]).sum() / (raster(b["c"][i], b["r"][i]) | b["mask"][i]).sum()
            for i in range(32)]
    assert np.mean(ious) > 0.95


def test_rada_bounded_influence_and_alignment():
    """Numerical check of Proposition 4 (bounded influence and alignment)."""
    rng = np.random.default_rng(0)
    Pg = init_params(0)
    Ps = [{k: v + 0.01 * rng.normal(size=v.shape).astype(np.float32) for k, v in Pg.items()} for _ in range(6)]
    types, n = ["poly", "poly", "box", "box", "box", "box"], [100.0] * 6
    newP, rel = rada(Pg, Ps, n, types, beta=1.0)
    g = flat(Pg, CONTOUR)
    d_poly = ((flat(Ps[0], CONTOUR) - g) + (flat(Ps[1], CONTOUR) - g)) / 2
    d = flat(newP, CONTOUR) - g
    alpha_max = 400.0 / (200.0 + 400.0)
    assert float(jnp.dot(d, d_poly)) >= 0.0
    assert float(jnp.linalg.norm(d - d_poly)) <= 2 * alpha_max * float(jnp.linalg.norm(d_poly)) + 1e-6
    assert all(0.0 <= r <= 1.0 for r in rel.values())
