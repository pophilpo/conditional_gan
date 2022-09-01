#!/usr/bin/env python3

from models import Generator, Discriminator, initialize_weights
from config import (
    LEARNING_RATE,
    BATCH_SIZE,
    IMAGE_SIZE,
    CHANNELS_IMG,
    Z_DIM,
    EPOCHS,
    FEATURES_DISC,
    FEATURES_GEN,
)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


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
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(EPOCHS):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)

            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)

            fake = gen(noise)

            # Train Discriminator
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

            disk_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disk_fake, torch.zeros_like(disk_fake))
            loss_disc = (loss_disc_fake + loss_disc_real) / 2

            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator

            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                    Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
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

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


if __name__ == "__main__":
    main()
