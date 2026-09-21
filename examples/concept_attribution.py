"""Known-answer CAV, TCAV and target-centroid SAE clamping experiment."""

import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    SAEFeatureClamping,
    SignalBatch,
    TopKSAE,
    attribute,
    fit_sae_code_reference,
    sae_concept_profile,
    tcav_permutation_test,
)


class QuadraticRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.representation = nn.Identity()

    def forward(self, data):
        hidden = self.representation(data)
        return hidden.square().flatten(1).sum(1)


def reference_concepts():
    generator = torch.Generator().manual_seed(611)
    shared = torch.randn(100, 3, generator=generator)
    positive = torch.cat((torch.full((100, 1), 2.0), shared), dim=1)
    negative = torch.cat((torch.full((100, 1), -2.0), shared), dim=1)
    features = torch.stack((positive, negative), dim=1).reshape(200, 4)
    labels = torch.tensor([True, False] * 100)
    train = torch.zeros(200, dtype=torch.bool)
    train[:160] = True
    return positive, negative, features, labels, train, ~train


def identity_sae():
    sae = TopKSAE(4, 4, 4).eval()
    with torch.no_grad():
        sae.encoder_weight.copy_(torch.eye(4))
        sae.encoder_bias.zero_()
        sae.decoder_weight.copy_(torch.eye(4))
        sae.decoder_bias.zero_()
    return sae


def main():
    positive, negative, fit_features, labels, train, test = reference_concepts()
    generator = torch.Generator().manual_seed(997)
    evaluation = torch.cat(
        (
            torch.full((80, 1), 2.0),
            4 * torch.randn(80, 3, generator=generator),
        ),
        dim=1,
    )
    batch = SignalBatch(
        evaluation.reshape(80, 1, 1, 4),
        tuple(f"positive-{index}" for index in range(80)),
        ("C3",),
        200,
        "concept-attribution-known-answer",
    )
    lens = EEGLens(
        QuadraticRepresentation().eval(),
        Adapter([ActivationSite("representation", "representation")]),
        model_id="concept-attribution-known-answer",
    )
    gradients = attribute(
        lens,
        batch,
        lambda output, current: output,
        method="gradient",
        sites=("representation",),
    )
    tested = tcav_permutation_test(
        gradients,
        "representation",
        fit_features,
        labels,
        train_mask=train,
        test_mask=test,
        alpha=1e-3,
        permutations=31,
        seed=19,
    )

    sae = identity_sae()
    profile = sae_concept_profile(sae, positive, negative, tested.concept.direction)
    feature = int(torch.argmax(profile.decoder_alignment.abs()))
    target_reference = fit_sae_code_reference(sae, negative)
    clamped = lens.run_with_interventions(
        batch,
        interventions=(
            SAEFeatureClamping(
                "representation",
                sae,
                (feature,),
                target_reference,
            ),
        ),
    )
    if tested.observed.score != 1 or feature != 0:
        raise RuntimeError("Known-answer concept attribution failed")
    if not float(clamped.output.mean()) < float(gradients.objective.mean()):
        raise RuntimeError("Known-answer concept clamping failed")

    print(f"held-out CAV balanced accuracy: {tested.concept.validation_balanced_accuracy:.3f}")
    print(f"TCAV score: {tested.observed.score:.3f} (permutation p={tested.p_value:.3f})")
    print(
        f"top SAE feature: {feature} "
        f"(firing-rate difference={float(profile.firing_rate_difference[feature]):.3f})"
    )
    print(
        "mean objective after target-centroid clamp:",
        f"{float(gradients.objective.mean()):.3f} -> {float(clamped.output.mean()):.3f}",
    )


if __name__ == "__main__":
    main()
