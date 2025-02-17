<p align="center">
    <a href="assets/BRB_method_overview.png">
        <img src="assets/BRB_method_overview.png" alt="Results" width="70%"/>
    </a>
</p>

<div align="center">
<a href="https://www.python.org/doc/versions/">
      <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python Versions">
</a>
<a  href="https://github.com/instadeepai/Mava/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License" />
</a>
<a  href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style" />
</a>
<!-- <a href="https://zenodo.org/doi/10.5281/zenodo.10916257"><img src="https://zenodo.org/badge/758685996.svg" alt="DOI"></a> -->
</div>

<h2 align="center">
    <p>BRB: Breaking the Reclustering Barrier</p>
</h2>

## Introduction

This is the codebase accompanying our paper on "Breaking the Reclustering Barrier" (BRB). To summarize, BRB prevents early performance plateaus in centroid-based deep clustering by periodically applying a soft reset to the feature encoder with subsequent reclustering. This allows the model to escape local minima and continue learning. We show that BRB significantly improves the performance of centroid-based deep clustering algorithms on various datasets and tasks.

## Algorithms and data sets

### Implemented clustering algorithms

Our repo contains a BRB implementation on top of the following algorithms:

- [DEC](https://arxiv.org/pdf/1511.06335)
- [IDEC](https://www.ijcai.org/proceedings/2017/0243.pdf)
- [DCN](https://arxiv.org/pdf/1610.04794)

### Implemented data sets

Our repo contains training code for the following data sets:

- MNIST
- FashionMNIST
- KMNIST
- USPS
- GTSRB
- OPTDIGITS
- CIFAR10
- CIFAR100-20

## Pretrained Models

We provide our pretrained SimCLR ResNet-18 models for CIFAR10 and CIFAR100-20 in the table below.
The folders contain Pytorch weights for all 10 seeds used to generate the results in the paper.
For the hyperparameters, please refer to the paper or the `configs` folder.

| Dataset | Models |
| :--- | :--- |
| CIFAR10 | <http://e.pc.cd/eGjy6alK> |
| CIFAR100-20 | <http://e.pc.cd/Yrjy6alK> |

The k-Means accuracy of these models is reported in the [results section](#contrastive-learning).

## Installation instructions

### Pip

You can install the BRB package and dependencies via pip:

```bash
pip install -e .
```

### Conda

1. Clone the repo.
2. Install the environment

    ```bash
    conda env create -f environment.yml
    ```

### Troubleshooting

In case you get an error with `threadpoolctl` you need to reinstall it with

```bash
pip uninstall threadpoolctl && pip install threadpoolctl
```

## Usage

### Configuration

We use a hierarchical configuration based on [tyro](https://github.com/brentyi/tyro). This allows specification of values in the config as well as overwriting them via the CLI. Lastly, it provides a more enjoyable development experience by providing autocomplete. All configurations are stored in the `configs` folder.

### Single runs

The configuration for a single run is in `configs/base_config.py`. An example call overwriting a number of parameters is:

```bash
python train.py --experiment.track-wandb=False --pretrain_epochs=1 --brb.reset_interval=10 --brb.reset_interpolation_factor=0.9 --dc_algorithm="dec" --clustering_epochs=20 --brb.reset_weights=True --dc_optimizer.lr=0.0001 --dc_optimizer.weight_decay=0.1 --dc_optimizer.optimizer=adam --activation_fn="relu" --dataset_name="usps"
```

### Batched runs

A batch of experiments to be run in parallel can be configured with the runner config in `config/runner_config.py`. Per convention, the *runner will iterate over parameter lists that are given as tuple*. Configs for multiple experiments can be added to the `Configs` dictionary and run with:

```bash
python runner.py idec_cifar10 --experiment.track_wandb=False
```

Here `idec_cifar10` specifies the name of the configuration to run from the `Configs` dictionary. As is done above, it is still possible to override parameters from the CLI. The runner will recursively glob all files in the `config` folder to discover configurations.

### SLURM integration

We provide a script to run the experiments on a SLURM cluster that allows overriding of the runner configuration depending on the hardware setup. You can execute the script with:

```bash
submit_sbatch.py
```

## Adapting BRB to other algorithms

BRB consists of three components that must be implemented when using it with an arbitrary centroid-based clustering algorithm:

1. **Soft reset**  
  A mechanism to (partially) apply a soft reset to the network. Our code is provided in `src/deep/soft_reset.py`.
2. **Reclustering**  
  An algorithm for clustering the data after the reset. Our code uses k-means for reclustering because the centroid-based algorithms we use are based on k-means. However, this can be replaced with a clustering algorithm that is more suited to the application at hand. In `src/deep/_clustering_utils.py`, we implement the following clustering algorithms: random, k-means, k-means++-init, k-medoids, and expectation maximization.
3. **Momentum resets**  
  As last step of BRB, one has to reset the momentum terms for the centroids. Our code is provided [here](https://github.com/timoklein/clustering-plasticity/blob/main/src/deep/brb_reclustering.py#L52-L57).

Once these components are implemented, one can use BRB with any centroid-based clustering algorithm by periodically applying them.

The two most important hyperparameters of BRB are the reset interval $T$ and the reset interpolation factor $\alpha$. The reset interval determines the frequency with which BRB is applied, while the reset interpolation factor determines the strength of the network reset. For the feed forward autoencoder our default values for these hyperparameters, $T=20$ and $\alpha=0.8$, should provide a good starting point. For the ResNet18, we used $T=10$ and set $\alpha=0.7$ for the MLP encoder and $\alpha=0.9$ for the last ResNet block.

## Logging

Per default, our code will log various training metrics using [Weights & Biases](https://github.com/wandb/wandb). This allows to track experiments and compare results easily.
For paper-quality plots with exact numbers, one has to first download the data from the Weights & Biases server:

```bash
python wandb_downloader.py
```

The data is stored in three DataFrames: pretrain metrics, clustering metrics, and test metrics. These can then be used to generate plots.

The downloader is flexible and can be configured in multiple ways:

- [`HYPERPARAMS`](https://github.com/timoklein/clustering-plasticity/blob/b97b24778793ebb66ba1013dc2029b152cd6df03/wandb_downloader.py#L19-L78) is a set containing the hyperparameters that are downloaded with the train metrics. These allow for filtering and aggregating the data later.
- [`DownloadArgs`](https://github.com/timoklein/clustering-plasticity/blob/b97b24778793ebb66ba1013dc2029b152cd6df03/wandb_downloader.py#L82-L151) is a configuration file that specifies the wandb user, project, and metrics to download. It defaults to the currently logged in user.
- One can choose to download only runs that satisfy certain critera using the [`FILTERS`](https://github.com/timoklein/clustering-plasticity/blob/b97b24778793ebb66ba1013dc2029b152cd6df03/wandb_downloader.py#L353-L372) set. Details on how to configure these can be found in the downloader script.

> [!CAUTION]
>
> ### Expensive metrics
>
> Certain metrics are expensive to compute and will significantly slow down the code when logged. These are:
>
> - **Purity** (most significant slowdown)
> - Voronoi plots
> - Uncertainty plots

## Results

<p align="center">
    <img src="assets/BRB_overall_results.png" alt="Results" width="50%"/>
</p>

### Autoencoder results

Results for DEC and IDEC with and without BRB using a Feed Forward Autoencoder. The full results table can be found [here](assets/BRB_autoencoder_results.png).

<p align="center">
    <img src="assets/BRB_DEC_IDEC_relative_improvement_spider.png" alt="AE Results" width="70%"/>
</p>

### Contrastive Learning

Results for DEC, IDEC, and DCN with and without BRB using a ResNet18 encoder.

<p align="center">
    <img src="assets/BRB_contrastive_results_table.png" alt="AE Results" width="70%"/>
</p>

## Acknowledgements

Our code builds on [ClustPy](https://github.com/collinleiber/ClustPy), which provided us with implementations for the clustering algorithms. We modified their code to fit our needs and added the BRB method. For self-labeling we used the [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification/tree/master) repository and applied it to our models. We would like to thank the authors for their work.

## Citation

If you use our code or pretrained models for your research, please cite our paper:

```bibtex
@article{miklautz2024breaking,
  title={Breaking the Reclustering Barrier in Centroid-based Deep Clustering},
  author={Miklautz, Lukas and Klein, Timo and Sidak, Kevin and Leiber, Collin and Lang, Thomas and Shkabrii, Andrii and Tschiatschek, Sebastian and Plant, Claudia},
  journal={arXiv preprint arXiv:2411.02275},
  year={2024}
}
```
