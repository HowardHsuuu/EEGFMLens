# Migrating from a11 to a12

a12 adds methods without removing the a11 runtime API. Existing `SignalBatch`
construction remains valid; the new final `transforms` field defaults to an empty
tuple. Runs now include `metadata["signal_transforms"]`, which is empty for untouched
inputs.

New public modules provide:

- `spectral.py`: periodograms, band power, band scaling and trial-matched spectral patching;
- `probes.py` and `diagnostics.py`: held-out linear readouts and representation audits;
- `sae.py`: a clean-room Top-K SAE and residual-preserving feature interventions;
- `circuits.py`: source-to-mediator path patching with identity controls.

Trial-scope spectral functions require contiguous nonoverlapping patches. Use patch
scope when `patch_stride_samples` differs from patch length. Spectral patching never
resamples or silently aligns channels. Cross-covariance subspaces are deliberately
not called LEACE because they do not covariance-whiten the representation.
