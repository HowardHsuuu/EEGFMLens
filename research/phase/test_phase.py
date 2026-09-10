import numpy as np
from signals import at_distance, fft_error, make_variants, phase_draw, quality, rotated


def test_fft_energy_mean_and_local_matching_on_known_burst():
    rng = np.random.default_rng(4)
    t = np.arange(3000) / 200
    x = (
        np.sin(2 * np.pi * 13 * t) * np.exp(-(((t - 6.5) / 0.5) ** 2))
        + 0.2 * rng.normal(size=len(t))
    ).astype(np.float32)
    variants = make_variants(x, 5, 10, 44)
    for name in ["global_phase", "global_sigma_phase", "circular_shift"]:
        y = variants[name]
        assert fft_error(x, y) < 1e-6
        np.testing.assert_allclose(
            np.sum(x.astype(float) ** 2), np.sum(y.astype(float) ** 2), rtol=1e-6
        )
        np.testing.assert_allclose(x.mean(), y.mean(), atol=1e-7)
    e = variants["local_event"]
    o = variants["local_off"]
    np.testing.assert_array_equal(e[:1000], x[:1000])
    np.testing.assert_array_equal(e[1600:], x[1600:])
    np.testing.assert_allclose(np.linalg.norm(e - x), np.linalg.norm(o - x), rtol=1e-5)
    # Local spectrum preservation must not be mislabeled as global preservation.
    assert fft_error(x, e) > 1e-3
    sigma = variants["global_sigma_phase"]
    f = np.fft.rfftfreq(len(x), 1 / 200)
    outside = (f < 11) | (f > 16)
    np.testing.assert_allclose(
        np.fft.rfft(x)[outside], np.fft.rfft(sigma)[outside], atol=1e-5, rtol=1e-5
    )
    np.testing.assert_array_equal(np.sort(variants["circular_shift"]), np.sort(x))
    original = quality(x, x, np.array([[6, 1]]))
    changed = quality(x, variants["global_phase"], np.array([[6, 1]]))
    assert changed["sigma_event_fraction"] < original["sigma_event_fraction"]


def test_bisection_zero_and_known_distance():
    x = np.sin(np.arange(600) * 0.2)
    draw = phase_draw(x, 9)
    target = np.linalg.norm(rotated(x, draw, 1) - x) * 0.65
    np.testing.assert_allclose(np.linalg.norm(at_distance(x, draw, target) - x), target, rtol=1e-10)
    np.testing.assert_allclose(at_distance(x, draw, 0), x, atol=1e-12)


def test_subject_estimate_does_not_weight_subject_by_window_count():
    from summarize import estimate

    rows = [
        dict(subject=1, row=0, v=0),
        dict(subject=1, row=0, v=2),
        dict(subject=1, row=1, v=3),
        dict(subject=2, row=2, v=10),
    ]
    result = estimate(rows, lambda r: r["v"])
    assert result["subjects"] == {"1": 2.0, "2": 10.0}
    assert result["mean"] == 6
