import torch
import numpy as np
from args import args
from data.prep_dataloader import get_dataloaders
from data.dataloader import Data
from models.cvae import CVAE
from utils.loss_function import ELBO
from tqdm import tqdm
from utils.visualizer import visualize_z, save_as_image, plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
import os
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

OUT_PATH = f"saved_model/stages/{args.save_path}"
print(OUT_PATH)

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(OUT_PATH + "/val_output", exist_ok=True)
os.makedirs(OUT_PATH + "/latent", exist_ok=True)
os.makedirs(OUT_PATH + "/loss", exist_ok=True)

writer = SummaryWriter(log_dir=f"{OUT_PATH}/logs")

trans = transforms.Compose([transforms.ToTensor()])

train_loader, valid_loader, test_loader = get_dataloaders(Data)

print("Samples in Trainingset:", len(train_loader) * args.batch_size)
print("Samples in Validationset:", len(valid_loader) * args.batch_size)
print("Samples in Testingset:", len(test_loader.dataset))

criterion = ELBO()

model = CVAE(num_points=args.points, z_dim=args.latent_size, c_dim=args.num_conditions).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# scheduler = StepLR(optimizer, step_size=args.epochs // 4, gamma=0.5)
scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=30, min_lr=1e-6)
# scheduler = ExponentialLR(optimizer, gamma=0.999)

if args.finetune:
    print(20 * "-")
    print("Finetune Mode")
    print(20*"-")
    path = f"saved_model/stages/{args.save_path}/model.tar"
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for param in model.decoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # optimizer = torch.optim.Adam([{'params': model.encoder.parameters(), 'lr': 1e-3},{'params': model.decoder.parameters(), 'lr': 1e-6}])


BEST_LOSS = np.inf
BEST_EPOCH = 0

for epoch in tqdm(range(1, args.epochs + 1)):
    latent_z = []
    len_dataset = len(train_loader.dataset)
    tot_train_loss, tot_rec_loss, tot_lat_loss = 0, 0, 0
    model.train()
    for die, punch, part, bhf in train_loader:
        bs = part.size(0)
        hlf = bs // 2

        points = torch.cat((die[:hlf], punch[:hlf]), dim=0)
        conditions = torch.cat((torch.zeros(hlf), torch.ones(hlf)), dim=0)

        points = points.to(device)
        one_hot_encoded = torch.nn.functional.one_hot(conditions.long(), num_classes=2).to(device)
        part = part.to(device)

        x_pred, mean, log_var, z = model(points, one_hot_encoded)
        latent_z.append(z.detach().cpu().numpy())
        loss, rec_loss, lat_loss = criterion.calculate(x_pred, points, mean, log_var)

        if args.sum_mean == "sum":
            tot_train_loss += loss.item()
            tot_rec_loss += rec_loss.item()
            tot_lat_loss += lat_loss.item()
        else:
            tot_train_loss += loss.item() * bs
            tot_rec_loss += rec_loss.item() * bs
            tot_lat_loss += lat_loss.item() * bs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_train_loss = tot_train_loss / len_dataset
    mean_train_rec = tot_rec_loss / len_dataset
    mean_train_latent = tot_lat_loss / len_dataset

    len_dataset = len(valid_loader.dataset)
    model.eval()
    with torch.no_grad():
        tot_test_loss, tot_rec_loss, tot_lat_loss = 0, 0, 0
        for  die, punch, part, bhf in valid_loader:
            bs = part.size(0)
            hlf = bs // 2

            points = torch.cat([die[:hlf], punch[:hlf]], dim=0)
            conditions = torch.cat([torch.zeros(hlf), torch.ones(hlf)], dim=0)

            points = points.to(device)
            one_hot_encoded = torch.nn.functional.one_hot(conditions.long(), num_classes=2).to(device)
            part = part.to(device)

            x_pred, mean, log_var, z = model(points, one_hot_encoded)

            loss, rec_loss, lat_loss = criterion.calculate(x_pred, points, mean, log_var)

            if args.sum_mean == "sum":
                tot_test_loss += loss.item()
                tot_rec_loss += rec_loss.item()
                tot_lat_loss += lat_loss.item()
            else:
                tot_test_loss += loss.item() * bs
                tot_rec_loss += rec_loss.item() * bs
                tot_lat_loss += lat_loss.item() * bs

    latent_z = np.concatenate(latent_z, axis=0)

    mean_test_loss = tot_test_loss / len_dataset
    mean_test_rec = tot_rec_loss / len_dataset
    mean_test_latent = tot_lat_loss / len_dataset

    loss_comparison = mean_test_loss

    if loss_comparison < BEST_LOSS:

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_comparison,
        }, f"{OUT_PATH}/model.tar")

        BEST_LOSS = loss_comparison
        BEST_EPOCH = epoch

    print(f"Epoch: {epoch}/{args.epochs}, Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_test_loss:.6f} with Rec Loss: {mean_test_rec:.6f}, KLD: {mean_test_latent:.6f}, Best Loss: {BEST_LOSS:.6f} at Epoch: {BEST_EPOCH}")

    writer.add_scalar("Loss/Train", mean_train_loss, epoch)
    writer.add_scalar("Loss/Validation", mean_test_loss, epoch)
    writer.add_scalar("Loss/Reconstruction_Train", mean_train_rec, epoch)
    writer.add_scalar("Loss/KL_Train", mean_train_latent, epoch)
    writer.add_scalar("Loss/Reconstruction_Validation", mean_test_rec, epoch)
    writer.add_scalar("Loss/KL_Validation", mean_test_latent, epoch)

    if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
        z_path = f"{OUT_PATH}/latent/latent_space_{epoch}.png"
        visualize_z(latent_z, z_path)

        img = Image.open(z_path)
        img_tensor = trans(img)
        writer.add_image("Latent Space", img_tensor, epoch)

        pred_path = f"{OUT_PATH}/val_output/epoch_{epoch}_prediction.png"
        save_as_image(points_pred=x_pred[0].detach().cpu().numpy(), points_true=points[0].detach().cpu().numpy(),
                      filename=pred_path)

        pred_img = Image.open(pred_path)
        pred_img_tensor = trans(pred_img)
        writer.add_image("Prediction", pred_img_tensor, epoch)

    # scheduler.step()
    scheduler.step(mean_test_loss)

writer.close()

print("Finale Learning Rate:", optimizer.param_groups[0]["lr"])