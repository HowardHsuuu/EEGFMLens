# Pretraining exposure audit

Checked against primary papers on 2026-09-10. These sources describe reported training corpora, not an independently audited per-recording manifest of the downloaded weights.

| Model | Reported pretraining data relevant to this study | Allowed interpretation on EEGMMIDB |
| --- | --- | --- |
| CBraMod | TUEG; main paper §3.1 | Dataset-disjoint relative to the reported pretraining corpus; do not claim person-level identity exclusion independently verified |
| CSBrain | TUEG; §3.1 | Same qualification as CBraMod |
| LaBraM | Appendix D explicitly includes EEG Motor Movement/Imagery Dataset, 109 volunteers, 47.3 hours, for tokenizer and model pretraining | Dataset exposure is documented; our split only holds subjects out of downstream fitting, not pretraining |

Primary sources:

- CBraMod: https://proceedings.iclr.cc/paper_files/paper/2025/file/bbbd6d915cb90be21c1254a82d45cedd-Paper-Conference.pdf
- CSBrain: https://arxiv.org/html/2506.23075v1#S3.SS1
- LaBraM: https://arxiv.org/html/2405.18765v1#A4

The existing three-model intervention study remains useful for comparing downstream information dependence, but it cannot demonstrate pretraining-unseen generalization for all three. Do not pool the three as an exposure-controlled foundation-model benchmark or attribute differences solely to architecture. Exact LaBraM recording/subject membership in the downloaded checkpoint has not been verified; documented corpus inclusion is enough to disallow a blanket unseen-data claim.

For stronger pretraining-unseen claims across at least three models, add a model with a documented disjoint training corpus or a separate dataset excluded from all three reported corpora. That stronger result is not established by the current experiment. Keep the ongoing LaBraM results, explicitly labeled, instead of silently treating the overlap as absent.

Preprocessing also differs from the original training recipes: CSBrain reports 0.3–75 Hz bandpass, 60 Hz notch, 200 Hz resampling and amplitude scaling. Our current common-input pipeline intentionally omits bandpass/notch. Its conclusions are conditional on that input recipe; native recipe sensitivity remains to be checked.
