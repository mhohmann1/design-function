import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args

class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=args.num_conditions, latent_dim=args.latent_size, dim=3, hidden_dim=128, variational=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.variational = variational

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)

        self.mean_1 = nn.Linear(hidden_dim + self.c_dim, self.latent_dim)

        self.log_var_1 = nn.Linear(hidden_dim + self.c_dim, self.latent_dim)

        self.fc_out = nn.Linear(hidden_dim + self.c_dim, self.latent_dim)

        self.actvn = nn.ReLU()
        self.pool = self.maxpool

    def maxpool(self, x, dim=-1, keepdim=False):
        out, _ = x.max(dim=dim, keepdim=keepdim)
        return out


    def forward(self, p, c=None):
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        net = self.pool(net, dim=1)

        if c is not None:
            net = torch.cat([net, c], dim=1)

        if self.variational:

            mean = self.mean_1(self.actvn(net))

            log_var = self.log_var_1(self.actvn(net))

            return mean, log_var

        else:

            return self.fc_out(self.actvn(net))

class MLPDecoder(nn.Module):
    def __init__(self, z_dim, num_points, c_dim):
        super(MLPDecoder, self).__init__()
        self.num_points = num_points
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.z_dim + c_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.out = nn.Linear(512, self.num_points * 3)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        x = x.reshape(-1, self.num_points, 3)
        return x
