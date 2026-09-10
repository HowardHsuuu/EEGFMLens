# Roadmap

## Implemented through the local 0.1.0a6 alpha

- Installable MIT package with optional EEG/LaBraM dependencies.
- Eleven checkpoint-verified model families, with per-adapter geometry contracts and scoped cleanup. See [coverage](model-coverage.md). CBraMod/LaBraM support sensor/patch semantics; coverage differs by model.
- Paired replacement, zero and orthonormal-subspace ablation.
- Strict loaders, checksummed run bundles and paired metrics.
- Offline examples/tests, official checkpoint tests and real EEG evidence.
- Public channel/time patching sweeps with per-trial matching, stable random locations, identity/cleanup checks and grouped descriptive summaries. See [sweep validation](sweep-validation.md).
- Python 3.10/3.12 CI configuration; hosted execution is not claimed by local tests.

## Next milestones

1. Broaden hardware/dtype and input-configuration validation; add bounded cache storage and analysis utilities driven by user workflows.
2. Complete the running three-model [contrast-only versus sensor-projection comparison](../research/mi/CONTRAST_INTERVENTION.md). Preserve all fixed ranks, subject-wise effects and energy-matched controls. The original [held-out MI study](../research/mi/REPORT.md) is complete but does not establish a common mechanism; the exploratory recovery studies motivate distinguishing loss of a readout from loss of information.
3. Consider cross-model alignment and intervention transport only after reproducible, selective within-model effects are established. Current results do not meet that prerequisite. Any confirmatory extension needs unused evaluation data; the completed [spindle/N2 pilot](../research/spindle/RESULTS.md) and existing MI test are not new held-out samples.
4. Broaden BrainOmni beyond the verified tiny encode configuration; cross-modality preprocessing and studies remain research extensions.
5. Test functional QKV/attention and gradient interfaces independently before advertising native write access.

GitHub/PyPI publication and package-name availability have not been verified. Public release should review the exact distribution, attribution and validation artifacts. This alpha does not promise TransformerLens feature parity or established maintenance.

Current acceptance evidence and gaps are maintained in [public readiness](public-readiness.md). Historical version-specific test counts remain in the changelog and archived ledgers; they should not be summed or interpreted as current platform coverage.
