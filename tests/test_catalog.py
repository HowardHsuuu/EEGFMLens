from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from eegfmlens import connect, model_info, supported_models
from eegfmlens.errors import ValidationError

EXPECTED = {
    "cbramod": "CBraModAdapter",
    "labram": "LaBraMAdapter",
    "eegpt": "EEGPTAdapter",
    "biot": "BIOTAdapter",
    "bendr": "BENDRAdapter",
    "brainomni": "BrainOmniAdapter",
    "csbrain": "CSBrainAdapter",
    "neurorvq": "NeuroRVQAdapter",
    "signaljepa": "SignalJEPAAdapter",
    "diver": "DIVERAdapter",
    "steegformer": "STEEGFormerAdapter",
}


def test_catalog_has_one_complete_contract_per_verified_family():
    specs = supported_models()
    assert len(specs) == 11
    assert {spec.family for spec in specs} == set(EXPECTED)
    assert all(spec.display_name and spec.component and spec.input_contract for spec in specs)
    assert all(spec.upstream.startswith("https://github.com/") for spec in specs)
    assert all(spec.reference and spec.integrations for spec in specs)
    assert all("contract-ci" in spec.evidence for spec in specs)
    assert {spec.family for spec in specs if "pinned-source-ci" in spec.evidence} == {
        "cbramod",
        "labram",
        "eegpt",
        "biot",
        "csbrain",
        "diver",
        "steegformer",
    }
    for spec in specs:
        assert len(spec.variants) == len(set(spec.variants))
        assert spec.default_variant in spec.variants
        assert (
            spec.integrations[spec.variants.index(spec.default_variant)].adapter.__name__
            == EXPECTED[spec.family]
        )


def test_model_guide_covers_every_catalog_family():
    guide = (Path(__file__).resolve().parents[1] / "docs" / "models.md").read_text()
    for spec in supported_models():
        assert f"| `{spec.family}` |" in guide


@pytest.mark.parametrize(
    "alias,family",
    [
        ("DIVER-1", "diver"),
        ("ST_EEGFormer", "steegformer"),
        ("Signal-JEPA", "signaljepa"),
        ("CBraMod", "cbramod"),
    ],
)
def test_catalog_aliases_are_explicit(alias, family):
    assert model_info(alias).family == family


def test_catalog_rejects_unknown_names_and_options_before_adapter_construction():
    with pytest.raises(ValidationError, match="Unsupported model family"):
        model_info("future-model")
    with pytest.raises(ValidationError, match="missing required options"):
        connect(nn.Identity(), "biot")
    with pytest.raises(ValidationError, match="unexpected options"):
        connect(nn.Identity(), "eegpt", channels=("C3",))
    with pytest.raises(ValidationError, match="variant"):
        connect(nn.Identity(), "bendr", variant="classifier", channels=("C3",))


class TinyCBraMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Identity()
        block = nn.Module()
        block.self_attn_s = nn.Identity()
        block.self_attn_t = nn.Identity()
        block.dropout2 = nn.Identity()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([block])

    def forward(self, data):
        return data


def test_connect_records_family_and_describes_site_capabilities():
    model = TinyCBraMod().eval()
    lens = connect(model, "CBraMod", model_id="example")
    assert lens.manifest["family"] == "cbramod"
    assert lens.manifest["variant"] == "default"
    assert lens.manifest["native_model_class"].endswith("TinyCBraMod")
    assert lens.manifest["upstream"] == model_info("cbramod").upstream
    assert lens.manifest["catalog_evidence"] == model_info("cbramod").evidence
    capabilities = lens.capabilities()
    assert tuple(item.name for item in capabilities) == tuple(site.name for site in lens.sites())
    assert all(item.selectors == ("whole", "axis", "sensor", "patch") for item in capabilities)
    assert all(item.writable for item in capabilities)


def test_connect_rejects_non_module():
    with pytest.raises(ValidationError, match="torch.nn.Module"):
        connect(SimpleNamespace(), "cbramod")
