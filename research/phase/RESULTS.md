# Phase-surrogate experiment: result and limits

**Both tested frozen encoder + N2-head systems depend on information beyond the whole 15-second Fourier magnitude spectrum. This experiment does not establish that pretraining learned a spindle-specific temporal-morphology mechanism.** Event-token restoration sometimes moves the output toward its original value, but its advantage over matched location controls is not consistent. A negative specificity result is not evidence that the models cannot represent spindle morphology.

The completed run uses 141 manually annotated N2 windows from eight DREAMS excerpts, three deterministic surrogate replicates per window, five input conditions, and five observed layers. Twenty-four otherwise spindle-positive N2 windows failed the contained-event or guarded-off-event selection. Subject window counts are 23, 29, 3, 5, 27, 21, 12, and 21. All encoder parameters, subject-held-out heads, splits and hyperparameters came from the previous corrected experiment. No head was refit. Results concern selected N2 windows; prediction flips are not overall classification accuracy or clinical performance.

## What the controls actually preserve

The maximum whole-window FFT magnitude error is 1.11e-7 for full-band phase, 5.64e-8 for sigma-only phase, and 8.77e-8 for circular shift, below the 1e-6 float32 tolerance. Sigma-only phase leaves other frequency coefficients unchanged up to numerical error. The operations preserve total energy and mean numerically. They do not preserve each one-second patch's spectrum, local envelope, or all sample-distribution properties. Sensitivity therefore also admits explanations involving local spectral energy or ordinary temporal filtering.

Sigma-only phase reduces the fraction of sigma energy inside the original annotations by 0.2625 on average (subject-bootstrap 95% interval -0.2955 to -0.2314); this fraction decreases in a subject-weighted 98.5% of cases. Full-band phase decreases it in 99.6%. This measures loss of concentration at the annotated times, not definitive destruction of every spindle. Circular shift also reduces concentration at the original times while preserving the full waveform up to periodic relocation; envelope CV and the sample-value distribution are unchanged. Thus concentration alone is not a morphology detector.

Local event and off-event edits preserve the edited three-second segment FFT (maximum relative errors 1.74e-7 and 1.35e-7) and match absolute input perturbation L2. They **do not** preserve the complete input's spectrum: average unweighted relative magnitude-spectrum changes are about 0.264 and 0.271. The largest join jumps reach 23.5 and 30.5 original difference SDs. These conditions are confounded by boundary artifacts and global spectral changes.

## Fixed-head responses

Absolute clean-versus-surrogate N2 margin differences, divided by each head's training-fold margin SD; windows and replicates are averaged within subject, then subjects equally. Parentheses show descriptive 95% subject-bootstrap intervals. Different conditions have different input perturbation magnitudes except the explicitly matched local pair.

| Encoder | Full-band phase | Sigma-only phase | Circular shift |
|---|---:|---:|---:|
| CBraMod pretrained | 0.471 (0.399–0.551) | 0.254 (0.202–0.312) | 0.250 (0.206–0.311) |
| LaBraM pretrained | 0.958 (0.834–1.127) | 0.523 (0.453–0.588) | 0.665 (0.605–0.740) |
| CBraMod random encoder | 0.834 (0.756–0.916) | 0.457 (0.426–0.489) | 0.149 (0.122–0.186) |
| LaBraM random encoder | 0.872 (0.743–1.038) | 0.708 (0.584–0.858) | 0.024 (0.020–0.029) |

Sigma-only phase flips the pretrained CBraMod and LaBraM N2 decisions in subject-weighted 10.9% and 14.4% of cases. These are substantial functional responses despite preserved global spectral energy, not just tiny hidden-state numerical changes. However, random-weight encoders with their separately fitted, fixed staging heads also respond strongly. That control prevents attributing phase sensitivity itself to foundation pretraining. There is only one random initialization per architecture; no population-level pretraining comparison is established.

With equal local input perturbation norm, event-minus-off-event absolute margin change is -0.0462 (-0.1001 to -0.0068) for pretrained CBraMod and 0.0258 (-0.0047 to 0.0609) for pretrained LaBraM. There is no consistent cross-model event-specific input sensitivity.

## Per-patch representations and functional restoration

All 15 patch distances are stored for embedding and blocks 2, 5, 8, 11, for every input condition and all four encoders. Local edits produce their largest early changes near the edited ROI, while later changes spread. Token position is not an exclusive receptive field: patch embedding, normalization, convolution and attention can mix information. Relative activation distances across layers also have different denominators and are descriptive, not comparable causal importance scores.

Restoration benefit is `(abs(corrupt-clean margin) - abs(restored-clean margin)) / training margin SD`. Positive values mean movement toward the original output. All selected event patches use unscaled clean donor values. Off-event and random controls use the same patch count and actual activation-delta Frobenius norm. CLS is excluded from these selected LaBraM interventions.

Under sigma-only phase, event restoration has descriptive positive benefit at CBraMod block 8 (0.0476, interval 0.0041–0.0979), LaBraM embedding (0.0386, 0.0071–0.0667), and LaBraM block 11 (0.0553, 0.0185–0.0894). These demonstrate that intervening at these sites can affect the phase-induced output difference in these cases. They do not locate a unique spindle circuit: the matched location comparisons below do not support event specificity. All sites were analyzed; these individual intervals are exploratory and uncorrected for multiple comparisons.

| Model / site | Event benefit minus off-event benefit | Event benefit minus random-position benefit |
|---|---:|---:|
| cbramod / embedding.output | 0.0219 (-0.0290, 0.0781) | 0.0241 (-0.0233, 0.0783) |
| cbramod / blocks.2.output | -0.0010 (-0.0566, 0.0596) | 0.0203 (-0.0340, 0.0820) |
| cbramod / blocks.5.output | 0.0204 (-0.0466, 0.0976) | 0.0376 (-0.0271, 0.1173) |
| cbramod / blocks.8.output | 0.0321 (-0.0264, 0.1145) | 0.0463 (-0.0138, 0.1270) |
| cbramod / blocks.11.output | 0.0248 (-0.0221, 0.0938) | 0.0341 (-0.0236, 0.1293) |
| labram / embedding.output | -0.0080 (-0.0415, 0.0239) | -0.0063 (-0.0335, 0.0256) |
| labram / blocks.2.output | -0.0362 (-0.0673, -0.0062) | -0.0242 (-0.0649, 0.0153) |
| labram / blocks.5.output | -0.0034 (-0.0348, 0.0311) | -0.0030 (-0.0561, 0.0517) |
| labram / blocks.8.output | -0.0319 (-0.0717, 0.0074) | -0.0169 (-0.0507, 0.0191) |
| labram / blocks.11.output | -0.0274 (-0.0635, 0.0069) | -0.0244 (-0.0602, 0.0079) |

No sigma-only event-versus-control comparison has a positive 95% interval excluding zero. This does not prove equivalence or absence. The descriptive subset with control multipliers at most two does not supply positive event-specific evidence either. Sigma-only control multipliers exceed two in 2.74% (CBraMod) and 1.70% (LaBraM) of control interventions, with maxima 2.87 and 3.08.

Local-event perturbations are much more problematic: matching the event-delta norm requires multipliers above two in 98.8% and 99.6% of location controls, with maxima 176 and 652. These can be far outside plausible clean/corrupt activation values. Local restoration therefore cannot substantiate event specificity, even when the clean event replacement appears favorable. No zero-delta control was unavailable in this completed run; all actual matched norms passed their numerical assertions. The implementation handles an unavailable control explicitly for future inputs.

## Decision

The narrow global-spectrum-only explanation is contradicted for these frozen encoder/readout systems on these inputs. The stronger claim that EEG foundation pretraining learned a spindle-specific temporal-morphology mechanism remains **inconclusive / unsupported by this experiment**. Generic phase/time sensitivity, local spectral-energy changes, sample-distribution changes, position sensitivity and artificial local boundaries remain adequate alternatives. Pretrained-model sensitivity to a morphology-preserving circular shift further cautions against equating any temporal response with spindle recognition.

The experiment fulfills its discrimination goal by separating this narrow positive result from the unsupported mechanism interpretation. EEGLens now has an end-to-end paired-signal, per-patch, norm-matched intervention example with explicit control failure reporting. A future mechanism study would need better-matched physiological counterfactuals and independent data; that is not a result claimed here.

All numerical results, including other variants and every site, are in `results/summary.json`. Three figures show aggregate responses, event-aligned patch changes, and the first selected waveform example. Bootstrap intervals use 10,000 subject resamples and only eight subjects, including two with very few eligible windows. This is an exploratory follow-up to the earlier study, not independent confirmation.
