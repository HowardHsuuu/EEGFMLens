"""Independent native-coordinate checks shared by validation runners.

The caller must derive native axes from upstream code, not adapter.restore.
This helper currently supports single-invocation tensor outputs only.
"""

import torch

from eeglens import Ablation, Selection, SubspaceAblation
from eeglens.errors import ValidationError


def check_features(lens, batch, site, clean, native_forward, *, native_axis=-1):
    assert site.expected_calls == 1 and site.tensor_index is None
    current = clean.cache[site.name].tensor
    features = current.shape[-1]
    basis = torch.eye(features, device=current.device, dtype=current.dtype)[:, :2]
    for values in ((0.0, 0.0), (0.25, -0.5)):
        center = torch.zeros(features, device=current.device, dtype=current.dtype)
        center[:2] = center.new_tensor(values)
        edited = lens.run_with_interventions(
            batch, interventions=[SubspaceAblation(site.name, basis, center)], sites=(site.name,)
        )

        def native_edit(module, args, output):
            result = output.clone()
            axis = native_axis % result.ndim
            assert result.shape[axis] == features
            index = [slice(None)] * result.ndim
            index[axis] = slice(0, 2)
            view = result[tuple(index)]
            shape = [1] * result.ndim
            shape[axis] = 2
            view -= view - center[:2].reshape(shape)
            return result

        handle = lens.model.get_submodule(site.module_path).register_forward_hook(native_edit)
        try:
            with torch.no_grad():
                expected = native_forward()
        finally:
            handle.remove()
        torch.testing.assert_close(edited.output, expected, rtol=0, atol=0)
        observed = edited.cache[site.name].tensor
        torch.testing.assert_close(observed[..., 2:], current[..., 2:], rtol=0, atol=0)
        torch.testing.assert_close(
            observed[..., :2], center[:2].expand_as(observed[..., :2]), rtol=0, atol=1e-6
        )
    return dict(
        native_feature_axis=native_axis,
        partial_feature_erasure=True,
        centered_feature_erasure=True,
        untouched_features_exact=True,
    )


def check_rejected_selectors(lens, batch, site):
    for selection in (Selection(sensors=(batch.channels[0],)), Selection(patches=(0,))):
        try:
            lens.run_with_interventions(batch, interventions=[Ablation(site.name, selection)])
        except ValidationError as exc:
            assert "no sensor/patch selector" in str(exc)
        else:
            raise AssertionError(f"Unsupported physical selector accepted at {site.name}")
        assert all(not module._forward_hooks for module in lens.model.modules())
    return dict(unsupported_sensor_patch_rejected=True)
