"""Known-answer EEG-PRISM source-space attribution experiment."""

import torch
from torch import nn

from eegfmlens import Adapter, EEGLens, SignalBatch, attribute, source_attribution


class LinearObjective(nn.Module):
    def forward(self, data):
        return data.flatten(1).sum(1)


def main():
    source_delta = torch.tensor(
        [
            [[[1.0, -2.0, 3.0, 0.5]], [[-1.0, 4.0, 2.0, -0.5]]],
            [[[2.0, 1.0, -1.0, 3.0]], [[0.5, -2.0, 1.0, 2.0]]],
        ]
    )
    forward = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    sensor = torch.einsum("cm,bmpt->bcpt", forward, source_delta)
    batch = SignalBatch(
        sensor,
        ("trial-a", "trial-b"),
        ("C3", "C4"),
        64,
        "source-attribution-known-answer",
    )
    lens = EEGLens(LinearObjective().eval(), Adapter([]), model_id="linear-source-demo")
    input_result = attribute(
        lens,
        batch,
        lambda output, current: output,
        method="input_x_gradient",
    )
    sources = source_attribution(
        input_result,
        source_delta,
        forward,
        ("source-left", "source-right"),
    )

    torch.testing.assert_close(
        sources.sum_over_time().sum(dim=1),
        input_result.input_attribution.flatten(1).sum(dim=1),
    )
    torch.testing.assert_close(
        sources.reconstruction_rmse,
        torch.zeros_like(sources.reconstruction_rmse),
    )
    print("source contributions:", sources.sum_over_time().tolist())
    print("relative inverse reconstruction error:", sources.relative_reconstruction_error.tolist())
    print("attribution conservation error:", sources.conservation_error.tolist())


if __name__ == "__main__":
    main()
