# Completion audit, 2026-09-09

The goal was a discriminating, rerunnable experiment; a positive spindle mechanism was not required. The completed inference is beyond-whole-window-spectrum sensitivity, with spindle-specific interpretation unsupported.

| Requirement | Inspected evidence | Outcome |
|---|---|---|
| Human spindle events, valid physical EEG scale | Corrected parent manifest/inputs/readout hashes, annotation-file hashes, `UV-text-verified` recipe | 141 eligible windows across all eight subjects; 24 exclusions recorded |
| Spectrum-preserving paired signals | Actual float32 FFT assertions; independent full regeneration; `quality.json` | Whole-window and local constraints separately passed; local global-spectrum changes disclosed |
| Event and off-event controls of equal input strength | Every stored signal checked for unchanged exterior; paired input L2 relative mismatch <=6.12e-9 | Passed |
| Measured temporal-concentration changes | Annotated sigma-energy fraction, envelope CV, shift control and first selected waveform figure | Concentration changes verified; not assumed to prove spindle destruction |
| Per-patch model responses and fixed N2 heads | Four encoders, 423 cases, five variants, five sites, 15 finite distances per response row | 42,300 complete response rows; native clean pooled features match parent run |
| Intermediate event restoration with matched location controls | Two pretrained models, four non-shift variants, five sites, three positions; native execution assertions | 50,760 complete intervention/control rows; actual norms matched; no unavailable control in this run |
| Hook and identity correctness | Per-case identity, all-position final-block recovery, post-intervention cleanup; 26 tests including official checkpoints | Passed |
| Valid control interpretation | Multipliers, local join diagnostics, random encoders, shift, paired subject-level contrasts | Spindle specificity unsupported; local-event controls require extreme scaling |
| Reproducibility | Entire signal/QC regeneration exact; eight cases/model repeated independently | 400 response rows and 960 intervention rows exactly equal across repeats |
| Complete reporting | `RESULTS.md`, aggregate JSON, all sites and variants, figures and source entry points | Completed; descriptive intervals, no best-layer selection or clinical claim |

The primary run lives outside the repository at `research/eeglens_phase/run-v1` under the workspace. The repeat is `research/eeglens_phase/verification-v1`. Repository `results/` contains only aggregate statistics/figures and validation records, not raw EEG, model weights, readout weights or individual activation arrays. The single-case waveform figure remains in the local primary run.

Execution used local CPU, sequential model processes, and existing local checkpoints. No lab resources, paid instances, external messages, uploads, commits or pushes were used. The fresh-run CLI composes the same preparation, per-model evaluation, summary and plotting entry points that were executed for the reported run. The initial provenance record is retained; final analysis source fingerprints are recorded separately because reporting code was completed after model execution. Evaluation source fingerprints are checked against the initial record.

Verification scope is explicit: both full pretrained evaluations and random input-response evaluations completed; independent model repetition covers one case per subject, not the entire dataset. Random encoders use one initialization each. The 26-test pass includes core contracts, real-checkpoint loading/identity and the four new phase-analysis tests. The upstream LaBraM TorchScript deprecation warning does not prevent execution.
