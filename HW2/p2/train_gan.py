'''
train and test GAN model on airfoils
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt
from dataset import AirfoilDataset
from gan import Discriminator, Generator
from utils import *


from torch.autograd import Variable

def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    cuda = True if torch.cuda.is_available() else False

    # define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr_dis = 0.0005 # discriminator learning rate 0.0005
    lr_gen = 0.0005 # generator learning rate
    num_epochs = 60
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # build the model
    dis = Discriminator(input_dim=airfoil_dim).to(device)
    gen = Generator(latent_dim=latent_dim, airfoil_dim=airfoil_dim).to(device)
    print("Distrminator model:\n", dis)
    print("Generator model:\n", gen)

    # define your GAN loss function here
    # you may need to define your own GAN loss function/class
    # loss = ?
    adversarial_loss = torch.nn.BCELoss()

    # define optimizer for discriminator and generator separately
    optim_dis = Adam(dis.parameters(), lr=lr_dis)
    optim_gen = Adam(gen.parameters(), lr=lr_gen)
    lp_gen=list()
    lp_dis=list()
    # train the GAN model
    for epoch in range(num_epochs):
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            y_real = local_batch.to(device)

            # train discriminator
            valid = Variable(Tensor(y_real.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(y_real.size(0), 1).fill_(0.0), requires_grad=False)
#            print("valid_shape",valid.size())

            # calculate customized GAN loss for discriminator
            # enc_loss = loss(...)
            z = Variable(Tensor(np.random.normal(0, 1, (y_real.shape[0], latent_dim))))
            
            real_loss = adversarial_loss(dis(y_real), valid)
            fake_loss = adversarial_loss(dis(gen(z)), fake)
            loss_dis = (real_loss + fake_loss) / 2
            lp_dis.append(loss_dis)
            optim_dis.zero_grad()
            loss_dis.backward()
            optim_dis.step()

            # train generator

            # calculate customized GAN loss for generator
            # enc_loss = loss(...)
            z_2 = Variable(Tensor(np.random.normal(0, 1, (y_real.shape[0], latent_dim))))
            gen_imgs = gen(z_2)
            loss_gen = adversarial_loss(dis(gen_imgs), valid)
            lp_gen.append(loss_gen)
            optim_gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()

#             print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {}, Generator loss: {}".format(
                    epoch, num_epochs, n_batch, loss_dis.item(), loss_gen.item()))
    torch.save(dis.state_dict(),'p2_model_dis.ckpt')
    torch.save(gen.state_dict(),'p2_model_gen.ckpt')
     #Visualization of training los vs epoch
    plt.figure()
    plt.plot(lp_gen, color='blue', label="Generator Loss")
    plt.plot(lp_dis, color='red', label="Discriminator Loss")
    plt.title('Loss during training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    # test trained GAN model
    num_samples = 100
    # create random noise 
    noise = torch.randn((num_samples, latent_dim)).to(device)
    # generate airfoils
    gen_airfoils = gen(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot generated airfoils
    plot_airfoils(airfoil_x, gen_airfoils)


if __name__ == "__main__":
    main()

