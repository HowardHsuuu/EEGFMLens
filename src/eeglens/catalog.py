"""One public connection path for the verified EEG model families."""

from dataclasses import dataclass

from torch import nn

from .adapters.bendr import BENDRAdapter
from .adapters.brainomni import BrainOmniAdapter
from .adapters.cbramod import CBraModAdapter
from .adapters.continuous import BENDREncoderAdapter, BIOTAdapter
from .adapters.csbrain import CSBrainAdapter
from .adapters.diver import DIVERAdapter
from .adapters.eegpt import EEGPTAdapter
from .adapters.labram import LaBraMAdapter
from .adapters.neurorvq import NeuroRVQAdapter
from .adapters.signaljepa import SignalJEPAAdapter
from .adapters.steegformer import STEEGFormerAdapter
from .errors import ValidationError
from .model import EEGLens


@dataclass(frozen=True)
class IntegrationSpec:
    """Construction contract for one adapter variant."""

    variant: str
    adapter: type
    required_options: tuple[str, ...] = ()
    optional_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    """User-facing support boundary for one verified model family."""

    family: str
    display_name: str
    component: str
    input_contract: str
    physical_selection: str
    upstream: str
    reference: str
    integrations: tuple[IntegrationSpec, ...]
    default_variant: str = "default"
    caveat: str | None = None

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(item.variant for item in self.integrations)


_SPECS = (
    ModelSpec(
        "cbramod",
        "CBraMod",
        "pretrained encoder or reconstruction model",
        "200 Hz; 200-sample patches",
        "sensor and patch at every declared site",
        "https://github.com/wjq-learning/CBraMod",
        "b9e961003214326972c567eff390e75b0287e32a",
        (IntegrationSpec("default", CBraModAdapter),),
    ),
    ModelSpec(
        "labram",
        "LaBraM",
        "pretrained student encoder",
        "200 Hz; 200-sample patches; supported native channel names",
        "sensor and patch at every declared site",
        "https://github.com/935963004/LaBraM",
        "c431221e6cfd23dbfa9950e0180682fb322b0548",
        (IntegrationSpec("default", LaBraMAdapter, optional_options=("output",)),),
    ),
    ModelSpec(
        "eegpt",
        "EEGPT",
        "EEGTransformer encoder",
        "256 Hz; configured channels, windows and contiguous patch size",
        "whole activation or explicit raw axis",
        "https://github.com/BINE022/EEGPT",
        "a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705",
        (IntegrationSpec("default", EEGPTAdapter),),
    ),
    ModelSpec(
        "biot",
        "BIOT",
        "BIOTEncoder",
        "200 Hz; declared complete checkpoint channel vocabulary",
        "whole activation or explicit raw axis",
        "https://github.com/ycq091044/BIOT",
        "validated source SHA256 2bba53052a93a005033b27ef89f20037fd7894780ce9da130b823c7c73029e01",
        (IntegrationSpec("default", BIOTAdapter, required_options=("channels",)),),
    ),
    ModelSpec(
        "bendr",
        "BENDR",
        "encoder plus contextualizer; encoder-only variant available",
        "256 Hz; declared ordered model inputs; contiguous patches",
        "whole activation or explicit raw axis",
        "https://github.com/SPOClab-ca/BENDR",
        "ac918abaec111d15fcaa2a8fcd2bd3d8b0d81a10",
        (
            IntegrationSpec("full", BENDRAdapter, required_options=("channels",)),
            IntegrationSpec("encoder", BENDREncoderAdapter, required_options=("channels",)),
        ),
        default_variant="full",
    ),
    ModelSpec(
        "brainomni",
        "BrainOmni",
        "native encode path of the tiny checkpoint",
        "256 Hz; declared sensors, positions and sensor types; contiguous patches",
        "whole activation or explicit raw axis",
        "https://github.com/OpenTSLab/BrainOmni",
        "340d6b5aba886af76b217272cdb3251651e9bf16",
        (
            IntegrationSpec(
                "default",
                BrainOmniAdapter,
                required_options=("channels", "positions", "sensor_types"),
            ),
        ),
        caveat="Native evaluation uses dropout; pair RNG when comparing runs.",
    ),
    ModelSpec(
        "csbrain",
        "CSBrain",
        "pretrained encoder",
        "200 Hz; 200-sample patches; declared original sensor order",
        "sensor and patch at every declared site",
        "https://github.com/yuchen2199/CSBrain",
        "185aee55b24d0410a830df8dd08d03f675616998",
        (IntegrationSpec("default", CSBrainAdapter, required_options=("channels",)),),
    ),
    ModelSpec(
        "neurorvq",
        "NeuroRVQ",
        "raw four-branch pretrained backbone features",
        "200 Hz; contiguous 200-sample patches; checkpoint channel vocabulary",
        "sensor and patch at all branch sites",
        "https://github.com/KonstantinosBarmpas/NeuroRVQ",
        "926e770d9d16b6aa308404280fa0cc0211a6f9fb",
        (IntegrationSpec("default", NeuroRVQAdapter, required_options=("channel_vocabulary",)),),
    ),
    ModelSpec(
        "signaljepa",
        "SignalJEPA",
        "local and contextual encoder path",
        "model sampling rate and embedded channel order; contiguous patches",
        "whole activation or explicit raw axis",
        "https://github.com/braindecode/braindecode",
        "Braindecode 1.8.1; full model revision 51232ee0795a60e4378c17befe1e2ea5e94450c4",
        (IntegrationSpec("default", SignalJEPAAdapter),),
    ),
    ModelSpec(
        "diver",
        "DIVER-1",
        "EEG features or native time reconstruction",
        "500 Hz; contiguous 500-sample patches; declared xyz positions",
        "sensor and patch at embedding/final physical sites; raw axis internally",
        "https://github.com/DIVER-Project/DIVER-1",
        "fae4d7c5a58f2f795ce767939ad191d9c7ba49b8",
        (
            IntegrationSpec(
                "default",
                DIVERAdapter,
                required_options=("channels", "positions"),
                optional_options=("output",),
            ),
        ),
        caveat="Native evaluation uses dropout; pair RNG when comparing runs.",
    ),
    ModelSpec(
        "steegformer",
        "ST-EEGFormer",
        "small pretrained encoder forward_features path",
        "128 Hz; declared channels and official channel mapping; contiguous patches",
        "whole activation or explicit raw axis",
        "https://github.com/LiuyinYang1101/STEEGFormer",
        "542ee17918c3c2c36ba1d4ea02bedff5eb149370",
        (
            IntegrationSpec(
                "default",
                STEEGFormerAdapter,
                required_options=("channels", "channel_mapping"),
            ),
        ),
    ),
)

_BY_FAMILY = {spec.family: spec for spec in _SPECS}
_ALIASES = {
    "cbramod": "cbramod",
    "labram": "labram",
    "eegpt": "eegpt",
    "biot": "biot",
    "bendr": "bendr",
    "brainomni": "brainomni",
    "csbrain": "csbrain",
    "neurorvq": "neurorvq",
    "signaljepa": "signaljepa",
    "signal-jepa": "signaljepa",
    "diver": "diver",
    "diver-1": "diver",
    "steegformer": "steegformer",
    "st-eegformer": "steegformer",
}


def supported_models() -> tuple[ModelSpec, ...]:
    """Return the eleven checkpoint-backed public integration contracts."""

    return _SPECS


def model_info(family: str) -> ModelSpec:
    """Resolve a model family or documented alias to its integration contract."""

    if not isinstance(family, str) or not family.strip():
        raise ValidationError("Model family must be a nonempty string")
    key = family.strip().lower().replace("_", "-")
    try:
        return _BY_FAMILY[_ALIASES[key]]
    except KeyError as exc:
        raise ValidationError(
            f"Unsupported model family {family!r}; available: {tuple(_BY_FAMILY)}"
        ) from exc


def connect(model: nn.Module, family: str, *, variant: str | None = None, model_id=None, **options):
    """Wrap a loaded native model through its verified family adapter.

    This function does not construct a model, load a checkpoint or preprocess EEG.
    It standardizes adapter selection and checks model-specific option names.
    """

    if not isinstance(model, nn.Module):
        raise ValidationError("model must be a torch.nn.Module")
    spec = model_info(family)
    variant = spec.default_variant if variant is None else variant
    integrations = {item.variant: item for item in spec.integrations}
    try:
        integration = integrations[variant]
    except (KeyError, TypeError) as exc:
        raise ValidationError(
            f"Unsupported {spec.display_name} variant {variant!r}; available: {spec.variants}"
        ) from exc
    missing = tuple(name for name in integration.required_options if name not in options)
    allowed = set(integration.required_options + integration.optional_options)
    unexpected = tuple(sorted(set(options) - allowed))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required options: {missing}")
        if unexpected:
            details.append(f"unexpected options: {unexpected}")
        raise ValidationError(f"Invalid {spec.display_name} adapter options; " + "; ".join(details))
    adapter = integration.adapter(model, **options)
    lens = EEGLens(model, adapter, model_id=model_id)
    lens.manifest.update(
        {
            "family": spec.family,
            "display_name": spec.display_name,
            "variant": variant,
            "native_model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "upstream": spec.upstream,
            "integration_reference": spec.reference,
        }
    )
    return lens
