# Bundled source and release-archive audit

The archives contain native model code from CBraMod and LaBraM. Other EEG model adapters connect caller-supplied implementations; their external repositories and weights are not bundled. The original six-item audit below was introduced for a4; a8 additionally retains and checks the PyTorch notice described next.

## PyTorch helper correspondence and notice

CBraMod's bundled `criss_cross_transformer.py` contains three helpers with exact AST matches to installed PyTorch 2.6.0 `torch/nn/modules/transformer.py`: `_get_activation_fn`, `_get_seq_len`, and `_detect_is_causal_mask`. The reference file matches its installed wheel RECORD checksum. This establishes correspondence, not the original upstream derivation revision or equivalence of the full modified transformer. Evidence: `research/model_validation/results/pytorch-helper-provenance.json`.

a8 retains the complete [PyTorch v2.6.0 LICENSE](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE) as `src/eeglens/_vendor/PYTORCH_LICENSE`. The auditor verifies its recorded upstream SHA256 and exact inclusion in wheel and sdist. No model Python code was changed. `research/model_validation/results/vendor-audit-a8.json` records seven source/license checks and both archive inventories. This notice improves attribution; it does not resolve the remaining LaBraM derivation-chain review.

## Original pinned-source checks

a9 additionally retains `BEIT2_LICENSE`, `TIMM_LICENSE`, `DEIT_LICENSE` and `DINO_LICENSE` for the projects explicitly acknowledged by LaBraM. Their inspected reference revisions, URLs and hashes are in `research/model_validation/results/labram-upstream-provenance.json`; both archives pass all eleven source/license checks. The reference BEiT v2 file is now under `beit2/modeling_finetune.py`, while LaBraM's historical `beitv2` link returns 404. `_cfg`, `DropPath`, `Mlp`, `Block.forward` and `PatchEmbed.forward` have exact AST matches to the reference. Nested class/method matches are not independent evidence counts.

The references retain Microsoft, Ross Wightman and Facebook source credits in THIRD_PARTY_NOTICES. Root NOTICE requests returned 404 at the four recorded reference revisions; `beit2/NOTICE` also returned 404. This is a check of those paths, not proof that no other notice exists anywhere in each project. The exact historical derivation commits remain unknown, and reference snapshots are not relabeled as them. No model computation changed.

| Bundled item | Pinned upstream | Check |
| --- | --- | --- |
| CBraMod transformer | `b9e961003214326972c567eff390e75b0287e32a` | Exact bytes |
| CBraMod model | Same revision, `models/cbramod.py` | Exact AST after the documented relative import and removal of the `__main__` demonstration |
| CBraMod license | Same revision | Exact bytes; retained in both archives |
| LaBraM model | `c431221e6cfd23dbfa9950e0180682fb322b0548`, `modeling_finetune.py` | Exact bytes |
| LaBraM channels | Same revision, `utils.py:standard_1020` | Exact ordered literal values, without executing the training utility |
| LaBraM license | Same revision | Exact bytes; retained in both archives |

Both upstream license files identify MIT terms. The package-level LICENSE and THIRD_PARTY_NOTICES are also checked for exact inclusion in wheel and sdist. The auditor reads pinned Git objects using `git show`, rather than trusting an upstream working tree. It does not fetch repositories or execute upstream modules.

`tools/audit_vendor.py` compares the audited vendor files to both archives and rejects defined checkpoint, recording, credential-file and repository/cache member patterns. No banned member was found. This is a defined inventory check, not a general secret scanner. A negative check added an unrecorded `+ 1` to CBraMod's encoder output in a temporary source copy; the auditor rejected it before producing a report.

## Reproduce

From the parent workspace, with local upstream clones containing the pinned objects:

```bash
python eeglens/tools/audit_vendor.py \
  --upstream-repositories research/eeglens_survey_2026-09-09/repos \
  --wheel eeglens/dist/api-a4/eeglens-0.1.0a4-py3-none-any.whl \
  --sdist eeglens/dist/api-a4/eeglens-0.1.0a4.tar.gz \
  --output vendor-audit.json
```

Evidence: `research/model_validation/results/vendor-audit-a4.json`, including revisions, upstream/bundled hashes, archive digests and auditor hash. Rebuilding an archive requires rerunning the audit against the new artifact.

## Limits

This establishes source equivalence and notice retention, not blanket permission for all uses. Checkpoint and dataset terms remain separate; neither is distributed in these archives. LaBraM's retained source header acknowledges BEiT-v2, timm, DeiT and DINO. This audit does not reconstruct every upstream derivation or certify the full transitive dependency licensing chain. Those acknowledgments remain intact; the release-readiness ledger distinguishes this inventory from remaining provenance/release review.
