"""User-declared integration for native PyTorch models without core changes."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch import nn

from ..errors import ValidationError
from ..types import ActivationSite, SignalBatch
from .base import Adapter


@dataclass(frozen=True)
class ModuleInfo:
    """Structural discovery only: a module path does not establish EEG semantics."""

    path: str
    module_type: str
    direct_parameters: int


def inspect_modules(model: nn.Module) -> tuple[ModuleInfo, ...]:
    """List native hook candidates without executing the model or installing hooks."""
    return tuple(
        ModuleInfo(
            path, type(module).__name__, sum(p.numel() for p in module.parameters(recurse=False))
        )
        for path, module in model.named_modules()
    )


class GenericAdapter(Adapter):
    """Declare hook sites and the native call for a user-supplied model.

    ``forward`` receives ``(model, SignalBatch, **execution_kwargs)``. It owns
    input reshaping and model-specific arguments. ``validate`` optionally checks
    the model's input contract before any hooks are installed. Unknown activation
    geometry should use layout ``batch`` (whole-activation edits only).
    """

    def __init__(
        self,
        sites: Iterable[ActivationSite],
        *,
        forward: Callable | None = None,
        validate: Callable[[SignalBatch], None] | None = None,
    ):
        super().__init__(list(sites))
        if forward is not None and not callable(forward):
            raise ValidationError("forward must be callable")
        if validate is not None and not callable(validate):
            raise ValidationError("validate must be callable")
        self._forward = forward
        self._validate = validate

    def validate(self, batch):
        if self._validate is not None:
            self._validate(batch)

    def forward(self, model, batch, **kwargs):
        if self._forward is None:
            return super().forward(model, batch, **kwargs)
        return self._forward(model, batch, **kwargs)
