# A Data-Driven Approach for Automating the Design Process of Deep Drawing Tools @ NUMISHEET 25

> **Abstract:** *The deep drawing tool development process, from method planning and design of tools to tool try-out and final commissioning, is very time-consuming and requires extensive iterative manual effort, particularly during the try-out stage. To accelerate the entire process, integrating obtained knowledge from the tool try-out stage into the early design stage offers significant potential. Towards automating tool design, this paper proposes a data-driven approach using a generative neural network to predict active surfaces of deep drawing tools based on given deep drawn parts, laying the foundation for incorporating try-out knowledge. The model is trained on active tool surfaces and their corresponding deep drawn parts, including variation of geometrical parameters and process parameters in deep drawing simulation. The approach is evaluated using simulated data from deep drawing processes. The proposed solution demonstrates an advancement in automatically generating the active tool surfaces for both the punch and the die directly from the desired deep drawn parts.*
> # Content
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)

# Installation

```
conda create -n design-function
conda activate design-function
conda env update -n design-function --file environment.yml
```
# Dataset
```
Download the dataset: https://opara.zih.tu-dresden.de/handle/123456789/1526
Unzip the files to the directory: ./data/dataset
To preprocess the dataset, run: python prepData.py
```

# Training

```
python trainCVAE.py
```

# Evaluation

```
python evalCVAE.py
```