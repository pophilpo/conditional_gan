#!/usr/bin/env python3

from models import Generator, Critic, initialize_weights
from config import (
    LEARNING_RATE,
    BATCH_SIZE,
    IMAGE_SIZE,
    CHANNELS_IMG,
    Z_DIM,
    EPOCHS,
    FEATURES_DISC,
    FEATURES_GEN,
    CRITIC_ITERATIONS,
    LAMBDA_GP,
)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from utils import gradient_penaly


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root="dataset", train=True, transform=train_transforms, download=True
    )

    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)

    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)

    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = torch.optim.Adam(
        critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)
    )

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    for epoch in range(EPOCHS):
        for batch_idx, (real, _) in enumerate(dataloader):
            gen.train()
            critic.train()

            real = real.to(device)

            # Train Critic
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
                fake = gen(noise)

                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)

                gp = gradient_penaly(critic, real, fake, device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + (
                    LAMBDA_GP * gp
                )
                opt_critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator

            output = critic(fake).reshape(-1)
            loss_gen = -torch.mean(output)
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    plt.imshow(
                        img_grid_fake.cpu()
                        .data.clamp(0, 1)
                        .permute(0, 2, 1)
                        .contiguous()
                        .permute(2, 1, 0),
                        cmap=plt.cm.binary,
                    )
                    plt.savefig(f"result_images/{step}.jpg")

                    writer_real.add_scalar("Critic Loss", loss_critic, global_step=step)
                    writer_real.add_scalar("Generator Loss", loss_gen, global_step=step)
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


if __name__ == "__main__":
    main()
