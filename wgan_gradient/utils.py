import torch
import torch.nn as nn


def gradient_penaly(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    # Calculated critic score
    mixes_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixes_scores,
        grad_outputs=torch.ones_like(mixes_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient.view(gradient.size(0), -1)

    # Taking L2 norm
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
