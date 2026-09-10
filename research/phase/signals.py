"""Explicit Fourier constraints and deterministic event-region selection."""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import wasserstein_distance


def phase_draw(x, seed):
    spectrum = np.fft.rfft(np.asarray(x, dtype=float))
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(spectrum))
    theta[0] = 0
    if len(x) % 2 == 0:
        theta[-1] = 0
    return spectrum, theta


def rotated(x, draw, strength):
    spectrum, theta = draw
    return np.fft.irfft(spectrum * np.exp(1j * strength * theta), n=len(x))


def at_distance(x, draw, distance):
    maximum = np.linalg.norm(rotated(x, draw, 1) - x)
    if distance < 0 or distance > maximum * (1 + 1e-10):
        raise ValueError("Requested phase distance is unattainable")
    lo, hi = 0.0, 1.0
    for _ in range(55):
        mid = (lo + hi) / 2
        if np.linalg.norm(rotated(x, draw, mid) - x) < distance:
            lo = mid
        else:
            hi = mid
    return rotated(x, draw, (lo + hi) / 2)


def regions(events, window_start):
    relative = np.asarray(events, dtype=float).copy()
    relative[:, 0] -= window_start
    for onset, duration in relative:
        if duration < 0.5 or onset < 0 or onset + duration > 15:
            continue
        start = int(np.clip(np.floor(onset + duration / 2) - 1, 1, 11))
        if start <= onset and onset + duration <= start + 3:
            return start, relative
    return None, relative


def off_regions(events, event_start):
    candidates = []
    for start in range(1, 12):
        if max(start, event_start) < min(start + 3, event_start + 3):
            continue
        if not any(
            onset < start + 4 and onset + duration > start - 1 for onset, duration in events
        ):
            candidates.append(start)
    return candidates


def fft_error(x, y):
    a, b = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(y))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def sigma_envelope(x):
    spectrum = np.fft.rfft(x)
    frequency = np.fft.rfftfreq(len(x), 1 / 200)
    spectrum[(frequency < 11) | (frequency > 16)] = 0
    return abs(hilbert(np.fft.irfft(spectrum, n=len(x))))


def quality(x, y, events, edit_start=None):
    envelope = sigma_envelope(y)
    mask = np.zeros(len(x), dtype=bool)
    for onset, duration in events:
        mask[max(0, int(onset * 200)) : max(0, min(len(x), int((onset + duration) * 200)))] = True
    power = envelope**2
    out = dict(
        global_fft_error=fft_error(x, y),
        relative_input_delta=float(
            np.linalg.norm(y - x) / max(np.linalg.norm(x - x.mean()), 1e-12)
        ),
        mean_delta=float(y.mean() - x.mean()),
        energy_relative_error=float(abs(np.sum(y * y) - np.sum(x * x)) / max(np.sum(x * x), 1e-12)),
        sigma_event_fraction=float(power[mask].sum() / max(power.sum(), 1e-12)),
        sigma_envelope_cv=float(envelope.std() / max(envelope.mean(), 1e-12)),
        distribution_wasserstein_sd=float(wasserstein_distance(x, y) / max(x.std(), 1e-12)),
        peak_sd=float(max(abs(y)) / max(x.std(), 1e-12)),
    )
    if edit_start is not None:
        a, b = edit_start * 200, (edit_start + 3) * 200
        out["local_fft_error"] = fft_error(x[a:b], y[a:b])
        out["join_jump_sd"] = float(
            max(abs(y[a] - y[a - 1]), abs(y[b] - y[b - 1])) / max(np.diff(x).std(), 1e-12)
        )
        assert np.array_equal(x[:a], y[:a]) and np.array_equal(x[b:], y[b:])
    return out


def make_variants(x, event_start, off_start, seed):
    x = np.asarray(x, dtype=float)
    e = slice(event_start * 200, (event_start + 3) * 200)
    o = slice(off_start * 200, (off_start + 3) * 200)
    de, do = phase_draw(x[e], seed), phase_draw(x[o], seed + 1)
    distance = 0.8 * min(
        np.linalg.norm(rotated(x[e], de, 1) - x[e]), np.linalg.norm(rotated(x[o], do, 1) - x[o])
    )
    event, off = x.copy(), x.copy()
    event[e] = at_distance(x[e], de, distance)
    off[o] = at_distance(x[o], do, distance)
    draw = phase_draw(x, seed + 2)
    distance_global = min(
        np.linalg.norm(x - x.mean()), 0.9 * np.linalg.norm(rotated(x, draw, 1) - x)
    )
    global_phase = at_distance(x, draw, distance_global)
    sigma_draw = phase_draw(x, seed + 3)
    freq = np.fft.rfftfreq(len(x), 1 / 200)
    sigma_draw[1][(freq < 11) | (freq > 16)] = 0
    sigma_distance = 0.9 * np.linalg.norm(rotated(x, sigma_draw, 1) - x)
    sigma_phase = at_distance(x, sigma_draw, sigma_distance)
    # float32 is the actual model input; audit its numerical spectrum.
    values = {
        k: v.astype(np.float32)
        for k, v in dict(
            global_phase=global_phase,
            global_sigma_phase=sigma_phase,
            local_event=event,
            local_off=off,
            circular_shift=np.roll(x, 1000),
        ).items()
    }
    for name in ["global_phase", "global_sigma_phase", "circular_shift"]:
        assert fft_error(x, values[name]) < 1e-6
    assert fft_error(x[e], values["local_event"][e]) < 1e-6
    assert fft_error(x[o], values["local_off"][o]) < 1e-6
    np.testing.assert_allclose(
        np.linalg.norm(values["local_event"] - x),
        np.linalg.norm(values["local_off"] - x),
        rtol=1e-5,
        atol=1e-7,
    )
    return values
