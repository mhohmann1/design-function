import torch
import torch.nn as nn
from models.encoder_decoder import SimplePointnet, MLPDecoder

class CVAE(nn.Module):
    def __init__(self, num_points, z_dim, c_dim):
        super(CVAE, self).__init__()
        self.num_points = num_points
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.encoder = SimplePointnet(self.c_dim)
        self.decoder = MLPDecoder(self.z_dim, self.num_points, self.c_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def sample(self, size, c):
        z = torch.randn(size, self.z_dim).cuda()
        return self.decoder(z, c)

    def forward(self, x, c):
        mean, log_var = self.encoder(x, c)
        z = self.reparameterize(mean, log_var)
        x_pred = self.decoder(z, c)
        return x_pred, mean, log_var, z