"""Reusable end-to-end workflows built from the low-level intervention API."""

from dataclasses import replace
from typing import Any, Sequence

import torch

from .errors import ValidationError
from .model import EEGLens
from .sweep import SweepResult, SweepTarget, patching_sweep
from .types import SignalBatch


def _single_trial(batch: SignalBatch, index: int) -> SignalBatch:
    return replace(
        batch,
        data=batch.data[index : index + 1],
        trial_ids=(batch.trial_ids[index],),
    )


def restoration_sweep(
    lens: EEGLens,
    clean: SignalBatch,
    recipient: SignalBatch,
    *,
    sites: Sequence[str] | None = None,
    targets: Sequence[SweepTarget] | None = None,
    random_controls: bool = True,
    seed: int = 0,
    multiplier_warning: float = 2.0,
    recovery_site: str | None = None,
    execution_kwargs: dict[str, Any] | None = None,
) -> SweepResult:
    """Measure how activation patching restores a corrupted model output.

    The score is the negative output error relative to a separately executed
    clean reference for the same trial. A positive ``clean_error_reduction`` in
    the returned rows therefore means that the intervention moved the recipient
    output toward its clean counterpart. Outputs must be batch-first tensors.

    This workflow measures restoration under a declared corruption. It does not
    infer a task metric, physiological source, or causal mechanism by itself.
    """

    clean.__post_init__()
    recipient.__post_init__()
    kwargs = dict(execution_kwargs or {})
    references: dict[str, torch.Tensor] = {}
    for index, trial_id in enumerate(clean.trial_ids):
        output = lens.run_with_cache(_single_trial(clean, index), sites=[], **kwargs).output
        if not isinstance(output, torch.Tensor) or output.ndim < 1 or output.shape[0] != 1:
            raise ValidationError("Restoration workflow requires batch-first tensor outputs")
        if not output.is_floating_point() or not torch.isfinite(output).all():
            raise ValidationError("Restoration workflow requires finite floating outputs")
        references[trial_id] = output.detach().clone()

    def score(output: Any, batch: SignalBatch) -> torch.Tensor:
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim < 1
            or output.shape[0] != len(batch.trial_ids)
            or not output.is_floating_point()
            or not torch.isfinite(output).all()
        ):
            raise ValidationError("Restoration workflow requires finite batch-first tensor outputs")
        reference = torch.cat([references[trial_id] for trial_id in batch.trial_ids]).to(output)
        if output.shape != reference.shape:
            raise ValidationError("Native output shape changed during restoration workflow")
        error = torch.linalg.vector_norm((output - reference).reshape(output.shape[0], -1), dim=1)
        scale = torch.linalg.vector_norm(reference.reshape(reference.shape[0], -1), dim=1)
        if bool((scale < 1e-8).any()):
            raise ValidationError(
                "Restoration workflow cannot normalize a near-zero clean reference"
            )
        return -error / scale

    result = patching_sweep(
        lens,
        clean,
        recipient,
        score,
        sites=sites,
        targets=targets,
        random_controls=random_controls,
        seed=seed,
        multiplier_warning=multiplier_warning,
        recovery_site=recovery_site,
        execution_kwargs=kwargs,
    )
    result.metadata["workflow"] = {
        "name": "restoration_sweep",
        "score": "negative normalized L2 output error to separately executed clean trial",
        "interpretation": "positive clean_error_reduction moves output toward clean reference",
    }
    return result
