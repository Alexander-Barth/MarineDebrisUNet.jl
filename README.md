# UNet for detecting suspected marine debris


## Installation

* You need the julia programming language available at https://julialang.org/downloads/ (test code is tested with julia version 1.10.0)
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

## Source code


* `litter_classification_train.jl`: train the neural network
* `litter_classification_validate.jl`: validate a single trained neural network
* `litter_classification_post.jl`: post-process several validation statistics of different neural networks

## Trained weights

Weights of the trained network are available at:  https://dox.ulg.ac.be/index.php/s/cPCMw5rjeX5gwTI
