# Statistical Flow Matching (SFM)

By Chaoran Cheng, Oct 2nd, 2024

This is the official repo for the NeurIPS 2024 paper *Categorical Flow Matching on Statistical Manifolds* by Chaoran Cheng, Jiahan Li, Jian Peng, and Ge Liu. The paper is available at [arXiv](https://arxiv.org/abs/2405.16441).


<p align="center">
<img src="assets/swissroll.gif" height="400" alt="swissroll"/>
<img src="assets/bmnist.gif" height="400" alt="bmnist"/>
</p>


## Introduction

We introduce *statistical flow matching* (SFM) as a novel discrete generative framework on the manifold of parameterized probability measures inspired by information geometry. Using the Fisher-Rao metric, we obtain the intrinsic Riemannian geometry of the statistical manifold and propose a numerically stable FM algorithm on categorical data with a diffeomorphism. SFM enjoys multiple advantages thanks to its continuous and geometric formulation. Check out our paper for more details!
![geometry](assets/geo.png)


## Installation

#### Stand-alone installation
Want to incorporate SFM into your own project? Simply copy `models/categorical.py` into your project and import it as a module:
```python
import torch.nn as nn
from models.categorical import SphereCategoricalFlow


class DummyVF(nn.Module):
    def __init__(self, n_class=2, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_class, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_class)
        )

    def forward(self, xt, t):
        # xt is the noised data of shape (batch_size, *data_dim, n_class)
        # t is the timestep of shape (batch_size,) or a scalar
        return self.fc(xt)


sfm = SphereCategoricalFlow(
    DummyVF(2, 128), data_dims=2, n_class=2, ot=False
)
```

The vector field encoder should be an `nn.Module` instance that accepts the current noised data `xt`, timestep `t`, and any (optional) additional conditional arguments as input. See [Usage](#usage) for more details. Below are the requirements (you've probably already had them!):

- `1.10 <= torch <= 1.13` OR `torch >= 2.0` (recommended). For `torch <= 1.13`, the package `functorch` is used. Note that in some early versions, `functorch` is not automatically installed with PyTorch, and you may need to install it manually. For `torch >= 2.0`, `torch.func` is now shipped with PyTorch.
- `torchdiffeq, tqdm, numpy`.
- `scipy` (optional). Only required for the optimal transport during training with `ot=True`.


#### Installation with pytorch-lightning
With this installation, you can run all the experiments in the paper except for the promoter design task. You can use pytorch-lightning with multi-GPU training (on Text8). `torch >= 2.0` is required to run the DiT model or the nanoGPT model with Flash Attention. `jupyter` is not included in the environment. The environment was built with Python version of 3.10.14, PyTorch version of 2.4.1, CUDA version of 12.1, and PyTorch Lightning version of 2.4.0.

```bash
conda env create -f env_lightning.yml
```

#### Promoter design task (optional)
To run the promoter design experiments, first download the datasets and the pre-trained Sei model [here](https://doi.org/10.5281/zenodo.7943307) and put them under `data/promoter` (some extraction may be needed). Then install the additional dependencies:
```bash
pip install pytabix pyBigWig pyfaidx pandas
```


## Usage

Our implementation is designed to be flexible and easy to use. As demonstrated [above](#stand-alone-installation), you can easily incorporate SFM into your own project by defining a vector field encoder. An arbitrary number of conditional arguments can also be passed to the encoder. Below we document the main methods of the SFM class.

- Manifold operations
  - `dist(cls, p, q, eps=0.)` Calculate the geodesic distance $d_g(p,q)$ between two points on the manifold.
  - `exp(cls, p, u, eps=0.)` Calculate the exponential map $\exp_p(u)$.
  - `log(cls, p, q, eps=0.)` Calculate the logarithmic map $\log_p(q)$.
  - `interpolate(cls, p, q, t, eps=0.)` Calculate the interpolant between two points.
  - `vecfield(cls, p, q, t, eps=0.)` Calculate the interpolant and the vector field.
- Model operations
  - `forward(self, t, pt, *cond_args)` Forward pass of the model for predicting the vector field. The conditional arguments are assumed to have the first dimension as the batch dimension.
  - `get_loss(self, p, *cond_args)` Calculate the loss of the model given the target data.
- Sampling & NLL functions
  - `sample(self, method, n_sample, n_steps, device, *cond_args, return_traj=False)` Sample from the model using different methods.
  - `compute_nll(self, method, p1, n_steps=200, tmax=1., tmin=0., exact=False, verbose=False)` Calculate the NLL of given data.
  - `compute_elbo(self, method, p1, n_steps=200, tmax=0.995, verbose=False)` Calculate the ELBO for NLL of given one-hot data.

We also provide an implementation for the naive SFM that directly learns the vector field without the diffeomorphism in `SimpleCategoricalFlow` and a linear flow matching model that assumes a flat Euclidean geometry of the simplex in `LinearCategoricalFlow`. The interfaces are identical to `SphereCategoricalFlow`. More concrete examples can be found in [Notebook](#notebook).

To train the model with the provided training script (binarized MNIST as an example), you can use the following command:
```bash
python main.py configs/bmnist.yaml --savename bmnist
```
Most arguments in the config file are self-explanatory, and the config files for other tasks are provided under the `configs` directory. Feel free to modify them to suit your needs.
To train the model using multiple GPUs, make sure you have pytorch-lightning properly installed following the [instructions](#installation-with-pytorch-lightning) above, and run the following command:

```bash
python main_lightning.py configs/dit.yaml --savename text8_dit
```

Note that the DiT model for Text8 uses [flash attention](https://github.com/Dao-AILab/flash-attention) and is hardcoded for `bf16` training. A `torch >= 2.0` is required to run the model. Some older NVIDIA GPUs and older versions of CUDA drivers may not support `bf16` training.

## Notebook

We provide several notebooks to demonstrate the usage of SFM on different datasets. To run these notebooks, make sure Jupyter Notebook or JupyterLab is properly installed. Also, install `plotly` for interactive plots using `pip install plotly`.

#### `vis_simplex.ipynb`
This notebook provides the visualization of the Riemannian structure and the Euclidean structure of the probability simplex. Geodesic distances, exponential maps (geodesics), and logarithm maps (vector fields) are plotted.

#### `swissroll.ipynb`
In this notebook, we train SFM and LinearFM on the Swiss roll on simplex dataset and calculate the NLL of the training samples.

#### `eval_bmnist.ipynb`
In this notebook, we evaluate the trained SFM on the binary MNIST dataset. We calculate the FID of the generated samples and the NLL of the test data.



## Reference

if you find this repo useful, please consider citing our paper:
```bibtex
@inproceedings{cheng2024categorical,
  title={Categorical Flow Matching on Statistical Manifolds},
  author={Cheng, Chaoran and Li, Jiahan and Peng, Jian and Liu, Ge},
  booktitle={Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024},
  year={2024},
}
```
