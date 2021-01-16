import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, input_dim,hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        # define your feedforward pass
        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]
        return z_mu,z_var


class Decoder(nn.Module):
    def __init__(self, latent_dim,hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        self.linear = nn.Linear(latent_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # define your feedforward pass
        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]
        return predicted



class VAE(nn.Module):
    def __init__(self, airfoil_dim,hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(airfoil_dim,hidden_dim, latent_dim)
        self.dec = Decoder(latent_dim, hidden_dim,airfoil_dim)
    
    def forward(self, x):
        # define your feedforward pass
        z_mu, z_var = self.enc(x)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        predicted=self.dec(x_sample)
        return predicted, z_mu, z_var

    def decode(self, z):
        # given random noise z, generate airfoils
        return self.dec(z)

