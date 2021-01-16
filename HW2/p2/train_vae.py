'''
train and test VAE model on airfoils
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
from dataset import AirfoilDataset
from vae import VAE
from utils import *


def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr = 0.001      # learning rate
    num_epochs = 60 #30
    hidden_dim= 512#256
    # build the model
    vae = VAE(airfoil_dim=airfoil_dim,hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    print("VAE model:\n", vae)

    # define your loss function here
    # loss = ?
    vae.train()
    # train_loss=0
    # define optimizer for discriminator and generator separately
    optim = Adam(vae.parameters(), lr=lr)
    loss_plot=list()
    # train the VAE model
    for epoch in range(num_epochs):
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            y_real = local_batch.to(device)
            
            # train VAE

            # calculate customized VAE loss
            # loss = your_loss_func(...)

            optim.zero_grad()
            x_sample, z_mu, z_var = vae(y_real)
            recon_loss = F.binary_cross_entropy(x_sample, y_real, size_average=False)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var) 
            loss = recon_loss + kl_loss
            loss_plot.append(loss)
            loss.backward()
            # train_loss += loss.item()
            optim.step()

            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))
    torch.save(vae.state_dict(),'p2_model.ckpt')
    #Visualization of training los vs epoch
    plt.figure()
    plt.plot(loss_plot, color='blue', label="training")
    plt.title('Loss during training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    # test trained VAE model
    num_samples = 100

    # reconstuct airfoils
    real_airfoils = dataset.get_y()[:num_samples]
    recon_airfoils, __, __ = vae(torch.from_numpy(real_airfoils).to(device))
    if 'cuda' in device:
        recon_airfoils = recon_airfoils.detach().cpu().numpy()
    else:
        recon_airfoils = recon_airfoils.detach().numpy()
    
    # randomly synthesize airfoils
    noise = torch.randn((num_samples, latent_dim)).to(device)   # create random noise 
    gen_airfoils = vae.decode(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot real/reconstructed/synthesized airfoils
    plot_airfoils(airfoil_x, real_airfoils)
    plot_airfoils(airfoil_x, recon_airfoils)
    plot_airfoils(airfoil_x, gen_airfoils)
    

if __name__ == "__main__":
    main()