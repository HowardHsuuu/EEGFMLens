"""Known-answer LEACE intervention with a same-rank random control."""

import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    LEACEAblation,
    SignalBatch,
    SubspaceAblation,
    categorical_targets,
    fit_leace_eraser,
    fit_random_subspace_control,
)


def balanced_fixture(seed, samples_per_class):
    """Make class offsets independent of correlated background variation."""

    generator = torch.Generator().manual_seed(seed)
    mixing = torch.tensor(
        [
            [1.0, 0.8, 0.0, 0.0],
            [0.0, 1.4, 0.3, 0.0],
            [0.2, 0.0, 1.1, 0.6],
            [0.0, 0.0, 0.4, 0.9],
        ]
    )
    background = torch.randn(samples_per_class, 4, generator=generator) @ mixing
    offsets = torch.tensor(
        [
            [2.0, -0.5, 0.0, 0.8],
            [-1.0, 1.5, 0.5, -0.4],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    features = (background[:, None, :] + offsets[None, :, :]).reshape(-1, 4)
    labels = tuple(f"class-{index % 3}" for index in range(features.shape[0]))
    return features, categorical_targets(labels)


def cross_covariance_norm(features, concepts):
    x = features - features.mean(0)
    z = concepts - concepts.mean(0)
    return float(torch.linalg.matrix_norm(x.T @ z / (features.shape[0] - 1)))


def main():
    reference_features, reference_concepts = balanced_fixture(11, 100)
    held_out_features, held_out_concepts = balanced_fixture(29, 50)
    eraser = fit_leace_eraser(reference_features, reference_concepts)
    random_control = fit_random_subspace_control(
        reference_features,
        rank=eraser.rank,
        seed=17,
    )

    batch = SignalBatch(
        held_out_features.reshape(-1, 1, 1, 4),
        tuple(f"held-out-{index}" for index in range(held_out_features.shape[0])),
        ("C3",),
        200,
        "concept-erasure-known-answer",
    )
    lens = EEGLens(
        nn.Sequential(nn.Identity()).eval(),
        Adapter([ActivationSite("representation", "0")]),
        model_id="concept-erasure-known-answer",
    )
    erased = lens.run_with_interventions(
        batch,
        interventions=(LEACEAblation("representation", eraser),),
    ).output.reshape(-1, 4)
    controlled = lens.run_with_interventions(
        batch,
        interventions=(
            SubspaceAblation(
                "representation",
                random_control.basis,
                random_control.center,
            ),
        ),
    ).output.reshape(-1, 4)

    before = cross_covariance_norm(held_out_features, held_out_concepts)
    after = cross_covariance_norm(erased, held_out_concepts)
    random = cross_covariance_norm(controlled, held_out_concepts)
    if not after < before * 1e-5 or not random > after * 100:
        raise RuntimeError("Known-answer concept-erasure controls failed")
    print("fitted LEACE rank:", eraser.rank)
    print(f"held-out concept cross-covariance: {before:.6f} -> {after:.6f}")
    print(f"same-rank random control: {random:.6f} (seed={random_control.seed})")


if __name__ == "__main__":
    main()
