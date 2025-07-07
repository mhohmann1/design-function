import torch
from args import args
from pytorch3d.loss import chamfer_distance

class ELBO:
    def __init__(self, beta_factor=args.beta):
        self.beta_factor = beta_factor

    def reconstruction_loss(self, x_pred, x):
        if args.sum_mean == "sum":
            dist, _ = chamfer_distance(x_pred, x, batch_reduction="sum", point_reduction="sum")
            return dist
        else:
            dist, _ = chamfer_distance(x_pred, x)
            return dist

    def latent_loss(self, mean, log_var):
        if args.sum_mean == "sum":
            kld = 0.5 * torch.sum(-1 - log_var + mean.pow(2) + log_var.exp())
            return self.beta_factor * torch.sum(kld)
        else:
            kld = 0.5 * torch.sum(-1 - log_var + mean.pow(2) + log_var.exp(), dim=1, keepdim=True)
            return self.beta_factor * torch.mean(kld)

    def calculate(self, x_pred, x, mean=None, log_var=None):
        rec_loss = self.reconstruction_loss(x_pred, x)
        if mean is not None and log_var is not None:
            lat_loss = self.latent_loss(mean, log_var)
            elbo = rec_loss + lat_loss
            return elbo, rec_loss, lat_loss
        else:
            return rec_loss