# Create the mamba environment
mamba create -n brb python=3.11 pip wheel -y


# Install other mamba packages 
mamba install -n brb tqdm pandas scipy seaborn ipdb scikit-learn threadpoolctl ruff py-spy wandb \
                 jupyterlab einops nltk umap-learn pynvml tyro shortuuid scikit-learn-extra -y

# Install the correct torch version depending on args
if [ "$1" == "cpu" ]; then
    mamba install -n brb pytorch torchvision torchaudio cpuonly -c pytorch -y
elif [ "$1" == "gpu" ]; then
    mamba install -n brb pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
fi

mamba update --all -y

source activate brb
