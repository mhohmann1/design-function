import os
import torch
import numpy as np
from args import args
from data.prep_dataloader import get_dataloaders
from data.dataloader import Data
from models.cvae import CVAE
from utils.loss_function import ELBO
from tqdm import tqdm
from utils.visualizer import save_single_image, save_as_row
from metrics.evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets, emd_approx
import pandas as pd

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_loader, valid_loader, test_loader = get_dataloaders(Data)

print("Samples in Trainingset:", len(train_loader) * args.batch_size)
print("Samples in Validationset:", len(valid_loader) * args.batch_size)
print("Samples in Testingset:", len(test_loader.dataset))

criterion = ELBO()

SAVE_PATH = f"saved_model/stages/{args.save_path}"

os.makedirs(SAVE_PATH + "reconstructed", exist_ok=True)
os.makedirs(SAVE_PATH + "generated", exist_ok=True)
os.makedirs(SAVE_PATH + "interpolated", exist_ok=True)

model = CVAE(num_points=args.points, z_dim=args.latent_size, c_dim=args.num_conditions).to(device)
checkpoint = torch.load(SAVE_PATH + "model.tar", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Best Loss: {checkpoint['loss']:.6f}, at Epoch: {checkpoint['epoch']}")
model.eval()

global_min = np.array([0., 0., 0])
global_max = np.array([279.99963885 - 3.27888232e-04, 229.99968253 - 3.47547511e-04, 50.5 - 5.00000000e-01])

test_hist = []

latent_z = []
len_dataset = len(test_loader.dataset)

model.eval()
with torch.no_grad():
    tot_test_loss, tot_rec_loss, tot_lat_loss = 0, 0, 0
    tot_emd_loss = 0

    if args.eval_metrics:
        tot_jsd = 0
        mmd_cd_list = []
        cov_cd_list = []
        mmd_emd_list = []
        cov_emd_list = []

    for idx, (die, punch, part, bhf) in tqdm(enumerate(test_loader)):
        bs = part.size(0)


        points, conditions = die, torch.zeros(bs)

        # points, conditions = punch, torch.ones(bs)

        points = points.to(device)
        one_hot_encoded = torch.nn.functional.one_hot(conditions.long(), num_classes=2).to(device)
        part = part.to(device)

        x_pred, mean, log_var, z = model(points, one_hot_encoded)
        loss, rec_loss, lat_loss = criterion.calculate(x_pred, points, mean, log_var)

        loss_emd = emd_approx(x_pred, points)

        if args.sum_mean == "sum":
            tot_test_loss += loss.item()
            tot_emd_loss += torch.mean(loss_emd)
        else:
            tot_test_loss += loss.item() * bs
            tot_emd_loss += torch.mean(loss_emd) * bs

        tag = "matrize" if conditions[0] == 0. else "stempel" if conditions[0] == 1. else "none"

        for n in range(bs):
            if n < 4:
                save_as_row(points_pred=x_pred[n].detach().cpu().numpy(), points_true=points[n].detach().cpu().numpy(),
                                  filename=SAVE_PATH + f"reconstructed/{idx}_{n}_{tag}.png", min=global_min, max=global_max)

        z_rand = torch.randn(bs, args.latent_size).to(device)
        generated = model.decoder(z_rand, one_hot_encoded)

        if args.eval_metrics:
            tot_jsd += jsd_between_point_cloud_sets(generated.detach().cpu().numpy(), points.detach().cpu().numpy())

            batch_results = compute_all_metrics(generated, points, bs)
            batch_results = {k: v.cpu().item() for k, v in batch_results.items()}

            mmd_cd_list.append(batch_results['lgan_mmd-CD'])
            cov_cd_list.append(batch_results['lgan_cov-CD'])
            mmd_emd_list.append(batch_results['lgan_mmd-EMD'])
            cov_emd_list.append(batch_results['lgan_cov-EMD'])

        for n in range(bs):
            if n < 4:
                save_single_image(points_pred=generated[n].detach().cpu().numpy(), filename=SAVE_PATH + f"generated/{idx}_{n}_{tag}.png", min=global_min, max=global_max)

        t_steps = 3
        z1 = torch.randn(bs, args.latent_size).to(device)
        z2 = torch.randn(bs, args.latent_size).to(device)
        for t in range(t_steps + 1):
            alpha = t / t_steps
            z_interp = (1 - alpha) * z1 + alpha * z2

            generated_interp = model.decoder(z_interp, one_hot_encoded)

            save_single_image(points_pred=generated_interp[0].detach().cpu().numpy(), filename=SAVE_PATH + f"interpolated/{idx}_{tag}_interp_t{t}.png", min=global_min, max=global_max)

mean_cd_loss = tot_test_loss / (len_dataset)
mean_emd_loss = tot_emd_loss / (len_dataset)
print(f"CD: {mean_cd_loss:.6f}, EMD: {mean_emd_loss:.6f}")

if args.eval_metrics:
    mean_jsd = tot_jsd / len(test_loader)
    print(f"JSD: {mean_jsd:.6f}")

    df = pd.DataFrame({"MMD-CD": mmd_cd_list,"COV-CD": cov_cd_list,"MMD-EMD": mmd_emd_list,"COV-EMD": cov_emd_list})
    df.loc["Mean"] = df.mean()
    df.loc["Std Dev"] = df.std()
    print(df)

