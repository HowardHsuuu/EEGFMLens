# Related-work check — 2026-09-09

This is a targeted primary-source search, not an exhaustive systematic review. Search terms covered EEG foundation models with robustness, reference/montage invariance, nuisance subspaces, activation patching and representation repair. No publication-level novelty claim is established.

| Closest work | Existing contribution / overlap | Consequence for this study |
|---|---|---|
| [Širca et al., Beyond Accuracy (2026)](https://arxiv.org/html/2605.17562v1) | Perturbation robustness, attribution under corruption, block-wise probing and pooling effects across EEG FMs | Generic robustness + layer plots is insufficient differentiation; preserve token-level probing and distinguish zero-padding from channel removal |
| [Dou et al., INCEPT (2026)](https://arxiv.org/html/2608.24597v1) | Invariance-oriented pretraining, multi-view representations and transfer evaluation | Do not claim invariant EEG representations are a new objective; ask what an existing frozen model functionally preserves |
| [DARE-EEG (2026)](https://arxiv.org/html/2605.18298v1) | Mask/anchor alignment and mask invariance in EEG pretraining | Missing-channel/mask consistency is already an explicit learning target |
| [Lehn-Schiøler et al., SAE interpretability (2026)](https://arxiv.org/html/2605.13930v1) | Concept-grounded sparse features, selective/off-target steering and spectral decoding | Activation manipulation and entanglement analysis alone are not a novel contribution |
| [Reliability and disease sensitivity are dissociable (2026)](https://www.biorxiv.org/content/10.64898/2026.08.19.745052v1.full) | Indexed excerpts describe frozen EEG FM reliability, representation geometry and disease-related recoverability; full-text fetch returned 403 | Distinguish generic stability from task utility; a claim that stable embeddings are useful is inadequate |
| [Montage-agnostic event segmentation (2026)](https://pubmed.ncbi.nlm.nih.gov/41997170/) | Frozen-backbone adaptation across montages/datasets with a lightweight input adapter | Merely adding an adapter for montage transfer is not the proposed gap; this entry was checked at abstract level |
| [Heimersheim & Nanda, activation patching (2024)](https://arxiv.org/abs/2404.15255) | Interpretation of patching evidence and metric pitfalls | Oracle recovery must not be mistaken for unique localization or a deployable repair |

Candidate distinction, explicitly an inference from this search: separate fixed-head failure from recoverable task information, then test whether a training-derived intermediate correction transfers to held-out subjects and unseen shift strengths without a clean test donor. The novelty would depend on this controlled mechanistic comparison and its empirical outcome; the current search does not prove no prior work has done it.

EEG reference changes require multichannel assumptions. [EEGLAB's reference guide](https://eeglab.org/tutorials/ConceptsGuide/rereferencing_background.html) explains the role of channel coverage in average referencing. The current single-central-channel prepared inputs are not sufficient for that experiment.

## Cross-task candidate follow-up

[What Do EEG Foundation Models Capture from Human Brain Signals?](https://arxiv.org/html/2605.11410v1) is direct overlap for cross-task feature-use maps. Its stated setup uses task-fine-tuned checkpoints and separately estimated erasers. See [NEXT_QUESTION.md](NEXT_QUESTION.md) for the narrower, still-unverified candidate of transporting an A-derived intervention within one frozen checkpoint to B without B-side intervention selection.

[The Identity Trap in EEG Foundation Models](https://arxiv.org/html/2606.06647v1) additionally packages frozen representation diagnostics, subject-axis erasure, spectral ablation and external-cohort evaluations. A generic diagnostic toolkit or identity-removal result also has direct overlap. This reinforces the need to specify the transferred object and excluded target-side selection; it does not independently establish the novelty of the candidate.
