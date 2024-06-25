"""
Main Script
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import train, generate, plot_loss, plot_D_output
import argparse

from models.gan import Generator, Discriminator

supported_models = [
    'gan',
]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GAN on FashionMNIST')
parser.add_argument('--model', type=str, default='gan', choices=supported_models, help='which GAN to use')
args = parser.parse_args()

# Configuration
DATA_PATH = 'data'
BATCH_SIZE = 64
INPUT_SHAPE = (1, 28, 28)
SAVE_PATH = os.path.join('results', args.model)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print('device: {}'.format(device))

# Data Loading & Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
dataset = datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model & Optimization
if args.model == 'gan':
    latent_dim = 100
    generator = Generator(latent_dim=latent_dim, output_shape=INPUT_SHAPE).to(device)
    discriminator = Discriminator(INPUT_SHAPE).to(device)
else:
    raise NotImplementedError

criterion = nn.BCELoss()
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
print(discriminator)
print(generator)

# Training
num_epochs = 50
loss_D_list, loss_G_list, D_x_list, D_G_z_list = train(
    discriminator, generator, data_loader, num_epochs, device, optimizer_D, optimizer_G, criterion,
    gen_img=True, save_path=SAVE_PATH
)

# Save Model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
torch.save(discriminator.state_dict(), os.path.join(SAVE_PATH, 'discriminator.pth'))
torch.save(generator.state_dict(), os.path.join(SAVE_PATH, 'generator.pth'))
print('GAN Model Saved')

# Visualization
plot_loss(loss_D_list, loss_G_list, save_path=SAVE_PATH)
plot_D_output(D_x_list, D_G_z_list, save_path=SAVE_PATH)
