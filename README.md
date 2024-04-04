# UNet for detecting suspected marine debris

This repo allows you to train and run a UNet to segment Sentinel 2 image to detect suspected marine debris. It is integrated into the 
[POS2IDON](https://github.com/AIRCentre/POS2IDON) pipeline from the [AIRCentre](https://github.com/AIRCentre/POS2IDON).


## Installation

* You need the julia programming language available at https://julialang.org/downloads/ (the code is tested with julia version 1.8.0 and 1.10.0 on Linux with an NVIDIA GPU)
* Clone this repo:

```bash
git clone https://github.com/Alexander-Barth/MarineDebrisUNet.jl
```

* Install all dependencies by issuing the following julia commands:

```julia
using Pkg
cd("MarineDebrisUNet.jl")
Pkg.activate(".")
Pkg.instantiate()
```


## Dataset


The environment variable `MARIDA_DATASET` should contain the directory with the
[Marine Debris Archive (MARIDA)](https://zenodo.org/records/5151941#.YfFZ_PXP30o).
If this environment variable is not set, the data set is expected to in `~/Data/MARIDA_dataset`.
The data set must be uncompressed and the training script will expect for example the file `train_X.txt` to be located at
`$MARIDA_DATASET/splits/train_X.txt`.


## Source code

The main scripts are in the `src` directory:

* `litter_classification_train.jl`: train the neural network
* `litter_classification_validate.jl`: validate a single trained neural network
* `litter_classification_post.jl`: post-process several validation statistics of different neural networks

## Trained weights

Weights of the trained network are available at:  https://dox.ulg.ac.be/index.php/s/cPCMw5rjeX5gwTI and are distributed in the [BSON](https://en.wikipedia.org/wiki/BSON) format.
