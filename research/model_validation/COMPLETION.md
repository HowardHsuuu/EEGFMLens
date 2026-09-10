# Expansion completion audit

Objective: connect as many reasonable EEG foundation encoders as practical, excluding NeuroLM-style language-model pipelines. The assessed scope includes the existing CBraMod/LaBraM/EEGPT/BIOT/BENDR families, priority BrainOmni/REVE/CSBrain/SignalJEPA additions, the missing BENDR contextualizer, and additional NeuroRVQ/EEGMamba candidates. This is not a claim to exhaust every published model or checkpoint size.

| Requirement | Inspected evidence | Outcome |
|---|---|---|
| Preserve existing five families | CBraMod/LaBraM checkpoint tests in the final wheel suite; current-runtime EEGPT, both BIOT checkpoints and BENDR encoder results | Passed; 68 native conditions rerun |
| Add BrainOmni | `results/brainomni-tiny.json`, exported adapter, official checkpoint strict load | 24 paired-RNG native conditions; tiny encode only |
| Add CSBrain | `results/csbrain.json`, exported adapter | 28 conditions plus original-sensor selection verification |
| Add NeuroRVQ | `results/neurorvq.json`, exported adapter and invocation tests | 104 conditions, branch isolation, reordered donor recovery; raw pretrained backbone features |
| Add SignalJEPA | `results/signaljepa.json`, `signaljepa-direct-source.json`, exported adapter | Full checkpoint strictly loaded; 20 conditions plus donor recovery passed in both standard-import and direct-source environments |
| Complete BENDR context coverage | `results/bendr-context.json`, exported BENDRAdapter | Both native components strictly loaded; 22 exact conditions and reordered donor recovery |
| Assess REVE | Official gated access returned 403; `reve-architecture-only.json` and experimental adapter | 92 architecture-only conditions; no pretrained claim or gate bypass |
| Assess EEGMamba | Official Mamba2/fused Triton configuration and runtime requirements documented in EXPANSION.md; experimental adapter | Native execution unavailable on local Mac; no CPU rewrite claimed equivalent and no unauthorized remote compute |
| Keep general integration extensible | GenericAdapter, structured output selectors, adapter-owned geometry, invocation-specific sites and tests | No model-name dispatch added to runtime |
| Package and regression checks | Final wheel installed to an isolated target, 42 tests passed, including original checkpoint tests; all Python sources byte-compared with working tree | Passed; hash in `results/wheel-validation.json` |
| Scope and provenance | Coverage table, runner/source/checkpoint hashes, synthetic-input limitations, BrainOmni RNG caveat and NeuroRVQ load exclusions | Recorded explicitly |
| Exclude NeuroLM | No NeuroLM adapter or language-model integration added | Matches user scope |

Nine model families are checkpoint-verified, with ten public model-specific adapters because BENDR exposes both encoder-only and complete-composition adapters. The additional native validation runners cover 266 site/batch conditions, excluding repeated runs and the separate CBraMod/LaBraM suite checks. REVE and EEGMamba remain experimental and are excluded from that verified count.

All download, environment preparation and validation sessions used for this expansion have terminated successfully after resolving the BENDR restoration-layout mismatch. Work used local CPU and publicly available or authorized checkpoint downloads; no lab resources, paid compute, publication or pushes were used. Scientific task performance, biological mechanism claims and physiological cross-modal input validation remain outside this integration goal.
