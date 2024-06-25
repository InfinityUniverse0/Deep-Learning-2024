"""
Utils Functions
"""

import os
import torch
import matplotlib.pyplot as plt
import time


def train(
        discriminator, generator, data_loader, num_epochs, device, optimizer_D, optimizer_G, criterion,
        gen_img=False, save_path=None
):
    """
    Train GAN

    :param discriminator: discriminator model
    :param generator: generator model
    :param data_loader: data loader
    :param num_epochs: number of epochs
    :param device: device
    :param optimizer_D: optimizer for discriminator
    :param optimizer_G: optimizer for generator
    :param criterion: loss function
    :param gen_img: whether to generate images for visual test (default: False)
    :param save_path: save path (default: None for display only)
    :return: loss_D_list, loss_G_list, D_x_list, D_G_z_list
    """
    discriminator.to(device)
    generator.to(device)

    loss_D_list = []
    loss_G_list = []
    D_x_list = []
    D_G_z_list = []
    z_test = torch.randn(8, generator.get_latent_dim()).to(device) if gen_img else None
    if gen_img and save_path is not None:
        if not os.path.exists(os.path.join(save_path, 'images')):
            os.makedirs(os.path.join(save_path, 'images'))
    for epoch in range(num_epochs):
        loss_D = 0.0
        loss_G = 0.0
        D_x = 0.0
        D_G_z = 0.0
        total_num = 0
        start_time = time.time()  # Start Time
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            discriminator.train()
            generator.eval()
            optimizer_D.zero_grad()
            outputs = discriminator(real_images)
            D_x += outputs.sum().item()
            loss_real = criterion(outputs, real_labels)
            loss_real.backward()

            z = torch.randn(batch_size, generator.get_latent_dim()).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())  # detach to avoid backpropagation to generator
            loss_fake = criterion(outputs, fake_labels)
            loss_fake.backward()
            optimizer_D.step()

            # Train Generator
            discriminator.eval()
            generator.train()
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, generator.get_latent_dim()).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            D_G_z += outputs.sum().item()
            loss_g = criterion(outputs, real_labels)
            loss_g.backward()
            optimizer_G.step()

            loss_D += (loss_real.item() + loss_fake.item()) * batch_size
            loss_G += loss_g.item() * batch_size
            total_num += batch_size
        end_time = time.time()  # End Time
        elapsed_time = end_time - start_time  # Elapsed Time
        loss_D /= total_num
        loss_D /= 2  # because we have two losses for real and fake
        loss_G /= total_num
        D_x /= total_num
        D_G_z /= total_num
        loss_D_list.append(loss_D)
        loss_G_list.append(loss_G)
        D_x_list.append(D_x)
        D_G_z_list.append(D_G_z)

        print('Epoch [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f}, Elapsed Time: {:.2f}s'.format(
            epoch + 1, num_epochs, loss_D, loss_G, D_x, D_G_z, elapsed_time
        ))

        if gen_img:
            generator.eval()
            images = generator(z_test)
            images = (images + 1) / 2  # denormalize to [0, 1]
            images = images.cpu().detach().numpy()
            images = images.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            plt.figure()
            for i in range(8):
                plt.subplot(2, 4, i + 1)
                plt.imshow(images[i], cmap='gray')  # set cmap to 'gray'
                plt.axis('off')
            if save_path is not None:
                plt.savefig(os.path.join(save_path, 'images', 'epoch_{}.png'.format(epoch + 1)))
            plt.show()

    return loss_D_list, loss_G_list, D_x_list, D_G_z_list


def generate(generator, num_images, device):
    """
    Generate Images

    :param generator: generator model
    :param num_images: number of images
    :param device: device
    :return: generated images
    """
    generator.to(device)
    generator.eval()
    z = torch.randn(num_images, generator.get_latent_dim()).to(device)
    images = generator(z)
    return images


def plot_loss(loss_D_list, loss_G_list, save_path=None):
    """
    Plot Loss Curve

    :param loss_D_list: loss D list
    :param loss_G_list: loss G list
    :param save_path: save path (default: None for display only)
    """
    plt.figure()
    plt.plot(range(1, len(loss_D_list) + 1), loss_D_list, label='Loss of D')
    plt.plot(range(1, len(loss_G_list) + 1), loss_G_list, label='Loss of G')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Loss Curve')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.show()


def plot_D_output(D_x_list, D_G_z_list, save_path=None):
    """
    Plot Discriminator Output

    :param D_x_list: D(x) list
    :param D_G_z_list: D(G(z)) list
    :param save_path: save path (default: None for display only)
    """
    plt.figure()
    plt.plot(range(1, len(D_x_list) + 1), D_x_list, label='D(x)')
    plt.plot(range(1, len(D_G_z_list) + 1), D_G_z_list, label='D(G(z))')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Output')
    plt.title('Discriminator Output')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'discriminator_output.png'))
    plt.show()
