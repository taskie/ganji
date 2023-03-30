# See: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import os
import random
import sys
from datetime import datetime
from typing import Any, Callable, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms

import ganji
import ganji.datasets
from ganji.project.model import Config


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class GanjiDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        config: Config,
        size: Optional[int] = 28,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # dummy argument
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.config = config
        self.train = train
        self.file_name = "dataset.tp"
        self.file_path = os.path.join(self.root, self.file_name)

        if self._check_legacy_exist():
            self.data = self._load_legacy_data()
            return

        self.data = self._load_data()

    def _check_legacy_exist(self):
        return os.path.exists(self.file_path)

    def _load_legacy_data(self):
        return torch.load(self.file_path)

    def _load_data(self):
        config = self.config
        codepoints = ganji.datasets.find_codepoints(config.codepoint_set)
        font_index = 0 if config.font_index is None else config.font_index
        density_quantiles = (config.density_quantile_min, config.density_quantile_max)

        if config.dataset_random_seed is not None:
            randomizer = random.Random(config.dataset_random_seed)
        else:
            randomizer = None

        data = ganji.datasets.load_data_for_gan(
            codepoints,
            config.font,
            config.unit,
            font_index=font_index,
            density_quantiles=density_quantiles,
            randomizer=randomizer,
        )

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        torch.save(data, self.file_path)

        return data

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img = self.data[index, :, :, 0]

        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, 1

    def __len__(self) -> int:
        return len(self.data)


class Generator(nn.Module):
    def __init__(self, *, ngpu=1, nc=1, nz=100, ngf=64, original=False) -> None:
        super().__init__()
        self.ngpu = ngpu
        if original:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        else:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 4 x 4
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 8 x 8
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 16 x 16
                nn.ConvTranspose2d(ngf, nc, 2, 2, 2, bias=False),
                nn.Tanh()
                # state size. (nc) x 28 x 28
            )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, *, ngpu=1, nc=1, ndf=64, original=False) -> None:
        super().__init__()
        self.ngpu = ngpu
        if original:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.main = nn.Sequential(
                # input is (nc) x 28 x 28
                nn.Conv2d(nc, ndf, 5, 2, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 12 x 12
                nn.Conv2d(ndf, ndf * 2, 5, 2, 0, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 4 x 4
                nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 2 x 2
                nn.Conv2d(ndf * 4, 1, 2, 1, 0, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, input):
        return self.main(input)


class DCGan:
    def __init__(self, device, *, dir=None, ngpu=1, compiled=False) -> None:
        self.device = device
        self.ngpu = ngpu
        self.compiled = compiled

        dir = dir if dir is not None else "."
        self.directory = dir
        self.config, _state = ganji.project.load_metadata(dir)

        self.original = False
        if self.original:
            self.image_size = 64
            self.ngf = 64
            self.ndf = 64
        else:
            self.image_size = 28
            self.ngf = 32
            self.ndf = 32

        self.workers = 2
        self.batch_size = 128
        self.nc = 1
        self.nz = 100
        self.lr = 0.0002
        self.beta1 = 0.5

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        mnist_mode = False
        if mnist_mode:
            self.dataset = datasets.MNIST("mnist/", train=True, download=True, transform=self.transform)
            self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        else:
            self.dataset = GanjiDataset(
                "ganji/", train=True, size=self.image_size, config=self.config, download=True, transform=self.transform
            )
            self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.netG = Generator(nc=self.nc, nz=self.nz, ngf=self.ngf, original=self.original).to(self.device)
        self.netD = Discriminator(nc=self.nc, ndf=self.ndf, original=self.original).to(self.device)

        if (self.device.type == "cuda") and (self.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

        self.criterion = nn.BCELoss()

        self.fixed_noise = None

        self.real_label = 1.0
        self.fake_label = 0.0

        self.optimizerD = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def load_state(self, epoch=None):
        self.config, self.state = ganji.project.load_metadata(self.directory)
        if epoch is None and self.state.epoch >= 1:
            self.load_models()
        elif epoch >= 1:
            self.load_models(epoch)
        else:
            self.init_models()
        if self.compiled:
            self.netG = torch.compile(self.netG)
            self.netD = torch.compile(self.netD)
        print(self.netG)
        print(self.netD)

    def init_models(self):
        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
        models_dir = os.path.join(self.directory, "models")
        os.makedirs(models_dir, exist_ok=True)
        fixed_noise_file = os.path.join(models_dir, "fixed_noise.pt")
        torch.save(self.fixed_noise, fixed_noise_file)

    def load_models(self, epoch=None):
        models_dir = os.path.join(self.directory, "models")
        suffix = f"_{epoch:06d}" if epoch is not None else ""
        netG_file = os.path.join(models_dir, f"netG{suffix}.pt")
        self.netG.load_state_dict(torch.load(netG_file))
        netD_file = os.path.join(models_dir, f"netD{suffix}.pt")
        self.netD.load_state_dict(torch.load(netD_file))
        fixed_noise_file = os.path.join(models_dir, "fixed_noise.pt")
        self.fixed_noise = torch.load(fixed_noise_file)

    def save_state(self, epoch):
        next_epoch = epoch + 1
        self.save_models()
        if next_epoch % 100 == 0:
            self.save_models(next_epoch)
        self.state.epoch = next_epoch
        ganji.project.dump_state(self.directory, self.state)

    def save_models(self, epoch=None):
        models_dir = os.path.join(self.directory, "models")
        suffix = f"_{epoch:06d}" if epoch is not None else ""
        netG_file = os.path.join(models_dir, f"netG{suffix}.pt")
        torch.save(self.netG.state_dict(), netG_file)
        netD_file = os.path.join(models_dir, f"netD{suffix}.pt")
        torch.save(self.netD.state_dict(), netD_file)

    @property
    def log_path(self):
        return os.path.join(self.directory, "log.h5")

    def train(self):
        self.load_state()
        start = self.state.epoch
        num_epochs = self.config.epoch_end

        if not os.path.exists(self.log_path):
            with h5py.File(self.log_path, "a") as f:
                f.create_dataset("epoch", (1,), dtype=np.int32)
                f.create_dataset("loss_D", (num_epochs + 1,), maxshape=(None,), dtype=np.float32)
                f.create_dataset("loss_G", (num_epochs + 1), maxshape=(None,), dtype=np.float32)
                f.create_dataset("D_x", (num_epochs + 1), maxshape=(None,), dtype=np.float32)
                f.create_dataset("D_G_z", (num_epochs + 1, 2), maxshape=(None, 2), dtype=np.float32)

        for epoch in range(start, num_epochs):
            self._train_one_epoch(num_epochs, epoch)

    def _train_one_epoch(self, num_epochs, epoch):
        self.state.epoch = epoch
        self.state.update_time = datetime.now().timestamp()

        for i, (data, _) in enumerate(self.dataloader):
            self._train_with_images(num_epochs, epoch, i, data)

        if (epoch + 1) % 1 == 0:
            self.save_state(epoch)
            self.save_imgs(epoch + 1)

    def _train_with_images(self, num_epochs, epoch, i, data):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        self.netD.zero_grad()
        real_cpu = data.to(self.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
        output = self.netD(real_cpu).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        fake = self.netG(noise)
        label.fill_(self.fake_label)
        output = self.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        self.optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        self.netG.zero_grad()
        label.fill_(self.real_label)
        output = self.netD(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()

        eprint(
            "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
            % (epoch, num_epochs, i, len(self.dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
        )

        if i == len(self.dataloader) - 1:
            with h5py.File(self.log_path, "a") as log_file:
                log_file["epoch"][0] = epoch + 1
                log_file["loss_D"][epoch] = errD.item()
                log_file["loss_G"][epoch] = errG.item()
                log_file["D_x"][epoch] = D_x
                log_file["D_G_z"][epoch] = (D_G_z1, D_G_z2)

    def save_imgs(self, epoch, *, rows=None, columns=None, generate_mode=False, seed=None):
        r = 8 if rows is None else rows
        c = 8 if columns is None else columns

        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()

        # Invert and rescale images 0 - 255
        gen_imgs = 127.5 - 127.5 * fake.numpy()
        h, w = gen_imgs.shape[2:4]
        combined_image = np.zeros((r * h, c * w), dtype=gen_imgs.dtype)

        cnt = 0
        for i in range(r):
            for j in range(c):
                i_start = i * h
                i_end = (i + 1) * h
                j_start = j * w
                j_end = (j + 1) * w
                combined_image[i_start:i_end, j_start:j_end] = gen_imgs[cnt, 0, :, :]
                cnt += 1

        if generate_mode:
            image_path = os.path.join(self.directory, "generated", f"{epoch:06d}.png")
        else:
            image_path = os.path.join(self.directory, "training", f"{epoch:06d}.png")
        swp_path = image_path + ".swp"

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        Image.fromarray(combined_image.astype(np.uint8)).save(swp_path, "png")

        os.rename(swp_path, image_path)

    def generate(self, epoch):
        self.load_state(epoch)
        self.save_imgs(epoch, generate_mode=True)


def train(dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = 1 if device == "cuda" else 0
    dc_gan = DCGan(device, dir=dir, ngpu=ngpu)
    dc_gan.train()


def generate(dir, epoch=None, *, rows=None, columns=None, seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = 1 if device == "cuda" else 0
    dc_gan = DCGan(device, dir=dir, ngpu=ngpu)
    dc_gan.generate(epoch)


def log(dir):
    log_path = os.path.join(dir, "log.h5")
    with h5py.File(log_path, "r") as log_file:
        epoch = int(log_file["epoch"][0])
        loss_G_list = log_file["loss_G"]
        loss_D_list = log_file["loss_D"]
        D_x_list = log_file["D_x"]
        D_G_z_list = log_file["D_G_z"]
        print("epoch,loss_G,loss_D,D_x,D_G_z1,D_G_z2")
        for i in range(epoch):
            D_G_z = D_G_z_list[i]
            print(f"{i},{loss_G_list[i]},{loss_D_list[i]},{D_x_list[i]},{D_G_z[0]},{D_G_z[1]}")
