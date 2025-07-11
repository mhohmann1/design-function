import argparse

parser = argparse.ArgumentParser(description="Design Function for Deep Drawing")
parser.add_argument("--seed", default=42, help="Seed", type=int)
parser.add_argument("--path", default="data/dataset", help="Path of dataset.", type=str)
parser.add_argument("--epochs", default=250, help="Number of epochs.", type=int)
parser.add_argument("--batch_size", default=16, help="Number of batch size.", type=int)
parser.add_argument("--latent_size", default=128, help="Number of latent Z size.", type=int)
parser.add_argument("--save_at", default=10, help="Save model starting at X//3 epoch.", type=int)
parser.add_argument("--workers", default=4, help="Number of workers.", type=int)
parser.add_argument("--learning_rate", default=1e-3, help="Size of learning rate.", type=float)
parser.add_argument("--points", default=2_048, help="Number of points.", type=int)
parser.add_argument("--num_conditions", default=2, help="Number of conditions to concatenate.", type=int)
parser.add_argument("--beta", default=1e-4, help="Beta factor (weight) of KLD in Loss.", type=float)
parser.add_argument("--train_size", default=0.75, help="Ratio of Training Samples.", type=float)
parser.add_argument("--val_size", default=0.15, help="Ratio of Training Samples.", type=float)
parser.add_argument("--stages", action="store_true", help="Train model in Second Stage.")
parser.add_argument("--save_path", default="stages_model", help="Save path/dir.", type=str)
parser.add_argument("--sum_mean", default="mean", help="Sum or mean of loss function.", type=str)
parser.add_argument("--eval_metrics", action="store_true", help="Evaluate the model metrics (JSD,COV,MMD).")
parser.add_argument("--die", action="store_true", help="Evaluate die (default).")
parser.add_argument("--punch", action="store_true", help="Evaluate punch.")
args = parser.parse_args()