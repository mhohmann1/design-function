# A Data-Driven Approach for Automating the Design Process of Deep Drawing Tools @ NUMISHEET 25

> **Abstract:** *The deep drawing tool development process, from method planning and design of tools to tool try-out and final commissioning, is very time-consuming and requires extensive iterative manual effort, particularly during the try-out stage. To accelerate the entire process, integrating obtained knowledge from the tool try-out stage into the early design stage offers significant potential. Towards automating tool design, this paper proposes a data-driven approach using a generative neural network to predict active surfaces of deep drawing tools based on given deep drawn parts, laying the foundation for incorporating try-out knowledge. The model is trained on active tool surfaces and their corresponding deep drawn parts, including variation of geometrical parameters and process parameters in deep drawing simulation. The approach is evaluated using simulated data from deep drawing processes. The proposed solution demonstrates an advancement in automatically generating the active tool surfaces for both the punch and the die directly from the desired deep drawn parts.*
# Content
- [Installation](#installation)
- [Code overview](#code-overview)
- [Dataset](#dataset)
- [Training](#training)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Reproducibility and limitations](#reproducibility-and-limitations)

# Installation

Run all commands from the repository root. `environment.yml` includes Python 3.11, PyTorch 2.4.1, CUDA 11.8 and 12.4 packages.

```
conda create -n design-function
conda activate design-function
conda env update -n design-function --file environment.yml
```
# Code overview

- `args.py`: command-line parameters and defaults.
- `prepData.py`: raw-data preprocessing.
- `data/dataloader.py` / `data/prep_dataloader.py`: loading, normalization, and splitting.
- `models/cvae.py` / `models/encoder_decoder.py`: conditional variational autoencoder (CVAE).
- `trainCVAE.py` / `evalCVAE.py`: training / evaluation and test-set inference.
- `utils/loss_function.py` / `metrics/evaluation_metrics.py`: loss and metrics.

# Dataset

Download the dataset: 
```
https://doi.org/10.18419/DARUS-5158
```
Unzip the files to `data/dataset`. Each sample directory must contain `Data/01_Matrize_Viertel.npz`, `Data/03_Stempel_Viertel.npz`, and part files under `Data/Bauteil/*.npz`. NPZ files contain a `points` array of XYZ coordinates in millimeters.

If the NPZ files are not available, preprocess the raw STEP and XLSX files with:
```
python prepData.py
```

# Training

The loader samples 2,048 points per surface and applies dataset-specific offsets and scaling defined in `data/dataloader.py`. Samples are split into approximately 75% training, 15% validation, and 10% test data.

Stage 1 learns to reconstruct tool surfaces, conditioned on tool types:

```
python trainCVAE.py
```

Stage 2 loads the stage-1 checkpoint, freezes the decoder, and trains the encoder to predict tool surfaces from part point clouds:
```
python trainCVAE.py --stages
```

Both stages save the checkpoint with the lowest validation loss to `saved_model/stages/<save_path>/model.tar` (default: `stages_model`). Stage 2 overwrites this checkpoint, copy it beforehand to retain stage 1. TensorBoard logs and validation images are stored in the same run directory.

# Configuration

Parameters are defined in `args.py` and passed as command-line options, for example `python trainCVAE.py --epochs 250 --seed 42`. Use `python trainCVAE.py --help` for all options.

| Option | Default | Purpose |
| --- | --- | --- |
| `--path` | `data/dataset` | Dataset directory for training/evaluation |
| `--save_path` | `stages_model` | Run directory name |
| `--seed` | `42` | PyTorch/NumPy and split seed |
| `--epochs` | `250` | Epochs per stage |
| `--batch_size` | `16` | Batch size; use an even value for training |
| `--learning_rate` | `0.001` | Initial stage-1 learning rate |
| `--latent_size` | `128` | Latent dimension |
| `--beta` | `0.0001` | KL-divergence loss weight |
| `--points` | `2048` | Points per surface |

Learning rate, latent dimension, KL weight, and batch size are candidates for hyperparameter optimization; no automated search is included. Stage 2 currently fixes its initial learning rate at `0.001`. Model dimensions must match when loading a checkpoint.

# Evaluation

Evaluate stage-2 predictions on test-set parts (die by default; add `--punch` for punch):
```
python evalCVAE.py --stages --save_path stages_model/
```

The trailing slash in `--save_path` is currently required by the path concatenation in `evalCVAE.py`.

To additionally calculate Jensen–Shannon divergence (JSD), minimum matching distance (MMD), and coverage (COV) on randomly generated tool surfaces:
```
python evalCVAE.py --stages --eval_metrics --save_path stages_model/
```

Metrics are printed to the console; reconstructed, generated, and interpolated point-cloud images are saved in the run directory. MMD and COV use Chamfer distance (CD) and Earth Mover's distance (EMD).