"""Causal path tests composed from EEGFMLens activation replacement."""

from dataclasses import dataclass, replace
from typing import Any, Callable

import torch

from .errors import ValidationError
from .interventions import AxisSelection, Replacement, Selection
from .types import SignalBatch


@dataclass(frozen=True)
class PathTraceResult:
    """Per-trial scores from a source-to-mediator path-patching experiment."""

    clean_score: torch.Tensor
    recipient_score: torch.Tensor
    source_patched_score: torch.Tensor
    mediator_patched_score: torch.Tensor
    source_site: str
    mediator_site: str
    metadata: dict[str, Any]

    @property
    def source_effect(self) -> torch.Tensor:
        return self.source_patched_score - self.recipient_score

    @property
    def mediated_effect(self) -> torch.Tensor:
        return self.mediator_patched_score - self.recipient_score


def _scores(score: Callable, output: Any, batch: SignalBatch) -> torch.Tensor:
    with torch.no_grad():
        values = score(output, batch)
    if (
        not isinstance(values, torch.Tensor)
        or values.shape != (len(batch.trial_ids),)
        or not values.is_floating_point()
        or not torch.isfinite(values).all()
    ):
        raise ValidationError("Score must return one finite floating value per trial, shape [B]")
    return values.detach().clone()


def path_patch(
    lens,
    donor: SignalBatch,
    recipient: SignalBatch,
    score: Callable,
    *,
    source_site: str,
    mediator_site: str,
    source_selection: Selection | AxisSelection = Selection(),
    mediator_selection: Selection | AxisSelection = Selection(),
    execution_kwargs: dict[str, Any] | None = None,
) -> PathTraceResult:
    """Test whether a downstream state carries an induced source-site effect.

    The procedure patches donor state at ``source_site`` into the recipient and
    caches the resulting ``mediator_site`` state.  It then patches that induced
    mediator state into a fresh recipient run.  Identity patches at both sites
    are checked in score space.  The mediator must execute downstream of the
    source on the selected native path; the caller should use declared sites and
    a mechanistic hypothesis to choose the pair.
    """

    donor.__post_init__()
    recipient.__post_init__()
    if source_site == mediator_site:
        raise ValidationError("Source and mediator sites must differ")
    if set(donor.trial_ids) != set(recipient.trial_ids):
        raise ValidationError("Donor and recipient trial IDs must match exactly")
    for attribute in ("channels", "sampling_rate", "preprocessing_id", "unit", "stride"):
        if getattr(donor, attribute) != getattr(recipient, attribute):
            raise ValidationError(f"Donor and recipient {attribute} differ")
    if donor.data.shape[1:] != recipient.data.shape[1:]:
        raise ValidationError("Donor and recipient signal geometry differs")
    donor_order = [donor.trial_ids.index(trial) for trial in recipient.trial_ids]
    donor = replace(
        donor,
        data=donor.data[donor_order],
        trial_ids=recipient.trial_ids,
    )
    for site in (source_site, mediator_site):
        if not lens.adapter.require(site).writable:
            raise ValidationError("Path patching requires writable source and mediator sites")
    kwargs = dict(execution_kwargs or {})
    donor_run = lens.run_with_cache(donor, sites=[source_site], **kwargs)
    recipient_run = lens.run_with_cache(recipient, sites=[source_site, mediator_site], **kwargs)
    recipient_score = _scores(score, recipient_run.output, recipient)
    clean_score = _scores(score, donor_run.output, donor)

    source_identity = lens.run_with_interventions(
        recipient,
        interventions=[
            Replacement(source_site, recipient_run.cache[source_site], source_selection)
        ],
        **kwargs,
    )
    if not torch.allclose(
        _scores(score, source_identity.output, recipient), recipient_score, rtol=1e-5, atol=1e-6
    ):
        raise ValidationError("Source identity patch changed the score")

    induced = lens.run_with_interventions(
        recipient,
        interventions=[Replacement(source_site, donor_run.cache[source_site], source_selection)],
        sites=[mediator_site],
        **kwargs,
    )
    source_patched_score = _scores(score, induced.output, recipient)

    mediator_identity = lens.run_with_interventions(
        recipient,
        interventions=[
            Replacement(
                mediator_site,
                recipient_run.cache[mediator_site],
                mediator_selection,
            )
        ],
        **kwargs,
    )
    if not torch.allclose(
        _scores(score, mediator_identity.output, recipient),
        recipient_score,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValidationError("Mediator identity patch changed the score")

    mediated = lens.run_with_interventions(
        recipient,
        interventions=[
            Replacement(mediator_site, induced.cache[mediator_site], mediator_selection)
        ],
        **kwargs,
    )
    return PathTraceResult(
        clean_score,
        recipient_score,
        source_patched_score,
        _scores(score, mediated.output, recipient),
        source_site,
        mediator_site,
        {
            "method": "source intervention followed by induced-mediator intervention",
            "identity_controls": "passed",
            "trial_alignment": "trial_id",
            "cache_stage": "post_intervention",
        },
    )
