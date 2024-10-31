import os
import warnings
from collections import defaultdict

import numpy as np
import torch
from tqdm import trange

from config.optimizer_config import PretrainOptimizerArgs
from src.deep._data_utils import get_dataloader
from src.deep._optimizer_utils import setup_optimizer
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from ._utils import split_views


class FullyConnectedBlock(torch.nn.Module):
    """
    Feed Forward Neural Network Block

    Parameters
    ----------
    layers : list
        list of the different layer sizes
    batch_norm : bool
        set True if you want to use torch.nn.BatchNorm1d (default: False)
    dropout : float
        set the amount of dropout you want to use (default: None)
    activation_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: None)
    bias : bool
        set False if you do not want to use a bias term in the linear layers (default: True)
    output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the last layer, if None then it will be linear (default: None)

    Attributes
    ----------
    block: torch.nn.Sequential
        feed forward neural network
    """

    def __init__(
        self,
        layers: list,
        batch_norm: bool = False,
        dropout: float = None,
        activation_fn: torch.nn.Module = None,
        bias: bool = True,
        output_fn: torch.nn.Module = None,
        additional_last_linear_layer: bool = False,
    ):
        super().__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.bias = bias
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        fc_block_list = []
        for i in range(len(layers) - 1):
            fc_block_list.append(torch.nn.Linear(layers[i], layers[i + 1], bias=self.bias))
            if self.batch_norm and i < len(layers) - 2:
                fc_block_list.append(torch.nn.BatchNorm1d(layers[i + 1]))
            if self.dropout is not None:
                fc_block_list.append(torch.nn.Dropout(self.dropout))
            if self.activation_fn is not None:
                # last layer is handled differently
                if i != len(layers) - 2:
                    fc_block_list.append(activation_fn())
                else:
                    if self.output_fn is not None:
                        fc_block_list.append(self.output_fn())
        if additional_last_linear_layer:
            fc_block_list += [activation_fn()]
            fc_block_list += [torch.nn.Linear(layers[-1], layers[-1], bias=self.bias)]
        self.block = torch.nn.Sequential(*fc_block_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass a sample through the FullyConnectedBlock.

        Parameters
        ----------
        x : torch.Tensor
            the sample

        Returns
        -------
        forwarded : torch.Tensor
            The passed sample.
        """
        forwarded = self.block(x)
        return forwarded


class _AbstractAutoencoder(torch.nn.Module):
    """
    An abstract autoencoder class that can be used by other autoencoder implementations.
    Additionally, it allows the training with an contrastive loss (based on SimCLR)

    Parameters
    ----------
    reusable : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)
    use_contrastive_loss : bool
        If set to true, contrastive loss will be used instead of reconstruction loss
    separate_cluster_head: bool (default: False)
        If set to True, projector will be added to output of resnet instead of the output of the MLP cluster head.
        NOTE: Works currently only for Resnet architecture

    Attributes
    ----------
    fitted  : bool
        indicates whether the autoencoder is already fitted
    reusable : bool
        indicates whether the autoencoder should be reused by multiple deep clustering algorithms
    """

    def __init__(
        self,
        reusable: bool = True,
        use_contrastive_loss: bool = False,
        separate_cluster_head: bool = False,
    ):
        super().__init__()
        self.fitted = False
        self.reusable = reusable
        self.separate_cluster_head = separate_cluster_head
        self.use_contrastive_loss = use_contrastive_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for an encode function of an autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points

        Returns
        -------
        x : torch.Tensor
            should return the embedded data point
        """
        return x

    def project(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for a project function of the projector of a contrastive method.

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        projected : torch.Tensor
            should return the projected of embedded
        """
        return embedded

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for a decode function of an autoencoder.

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        decoded : torch.Tensor
            should return the reconstruction of embedded
        """
        return embedded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies both the encode and decode function.
        The forward function is automatically called if we call self(x).

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of embedded points

        Returns
        -------
        reconstruction : torch.Tensor
            returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction

    def loss(
        self, batch: list, loss_fn: torch.nn.modules.loss._Loss, device: torch.device
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_contrastive_loss:
            return self.contrastive_loss(batch=batch, loss_fn=loss_fn, device=device)
        else:
            return self.reconstruction_loss(batch=batch, loss_fn=loss_fn, device=device)

    def contrastive_loss(
        self,
        batch: list,
        loss_fn: torch.nn.modules.loss._Loss,
        device: torch.device,
        use_reconstruction: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the contrastive loss of a single batch of data.
        Assumes dataloader with augmentation (returning two views)

        Parameters
        ----------
        batch : list
            the different parts of a dataloader (id, samples, ...)
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction
        device : torch.device
            device to be trained on
        use_reconstruction : bool
            specify if additional reconstruction loss should be computed

        Returns
        -------
        loss : tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]
            a dictionary containing total loss, loss components and other metrics,
            the embedded input sample,
            the reconstructed input sample
        """
        assert isinstance(batch, list), "batch must come from a dataloader and therefore be of type list"
        if len(batch) != 3:
            raise ValueError(
                f"Minibatch must have 3 entries: indices, aug_sample, orig_sample (2nd view), but has {len(batch)} entries."
            )

        # Assume dataloader with two batches for view 1 and view 2
        view1_batch = batch[1]
        view2_batch = batch[2]
        assert view1_batch.shape == view2_batch.shape
        batch_data = torch.concat([view1_batch, view2_batch]).to(device)
        if self.separate_cluster_head:
            # TODO: Make this more general such that it works for all architectures
            _intermediate_emb = self.conv_encoder(batch_data)
            projected = self.project(_intermediate_emb)
            embedded = self.fc_encoder(_intermediate_emb)
        else:
            embedded = self.encode(batch_data)
            projected = self.project(embedded)

        # contrastive loss
        projected1, projected2 = split_views(projected)
        clr_loss = self.contrastive_loss_fn([projected1, projected2])
        loss = clr_loss

        # optional: reconstruction loss
        rec_loss = torch.tensor(0.0).to(device)
        reconstructed = None
        if use_reconstruction:
            reconstructed = self.decode(embedded)
            rec_loss = loss_fn(reconstructed, batch_data)
            loss = clr_loss + rec_loss

        loss_dict = {
            "loss": loss,
            "reconstruction_loss": rec_loss,
            "contrastive_loss": clr_loss,
        }

        return loss_dict, embedded, reconstructed

    def reconstruction_loss(
        self, batch: list, loss_fn: torch.nn.modules.loss._Loss, device: torch.device
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the loss of a single batch of data.
        NOTE: If dataloader with augmentation is used, this loss will reconstruct both the augmented and original data

        Parameters
        ----------
        batch : list
            the different parts of a dataloader (id, samples, ...)
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction
        device : torch.device
            device to be trained on

        Returns
        -------
        loss : tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]
            a dictionary containing total loss, loss components and other metrics,
            the embedded input sample,
            the reconstructed input sample
        """
        assert isinstance(batch, list), "batch must come from a dataloader and therefore be of type list"
        if len(batch) == 3:
            # augmentation data loader is used
            # Assume dataloader with two batches
            view1_batch = batch[1]
            view2_batch = batch[2]
            assert view1_batch.shape == view2_batch.shape
            batch_data = torch.concat([view1_batch, view2_batch]).to(device)
        else:
            batch_data = batch[1].to(device)

        embedded = self.encode(batch_data)
        reconstructed = self.decode(embedded)
        loss = loss_fn(reconstructed, batch_data)

        loss_dict = {
            "loss": loss,
            "reconstruction_loss": loss,
            "contrastive_loss": torch.tensor(0.0).to(device),
        }

        return loss_dict, embedded, reconstructed

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.modules.loss._Loss,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Evaluates the autoencoder.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction
        device : torch.device
            device to be trained on (default: torch.device('cpu'))

        Returns
        -------
        loss: torch.Tensor
            returns the reconstruction loss of all samples in dataloader
        """
        with torch.no_grad():
            self.eval()
            loss = torch.tensor(0.0)
            for batch in dataloader:
                new_loss, _, _ = self.loss(batch, loss_fn, device)
                loss += new_loss
            loss /= len(dataloader)
        return loss

    def fit(
        self,
        n_epochs: int,
        optimizer_params: PretrainOptimizerArgs,
        batch_size: int = 128,
        data: np.ndarray = None,
        data_eval: np.ndarray = None,
        dataloader: torch.utils.data.DataLoader = None,
        evalloader: torch.utils.data.DataLoader = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_params: dict = {},
        model_path: str = None,
        print_step: int = 0,
        device: torch.device = torch.device("cpu"),
        wandb_run: Run | RunDisabled | None = None,
    ) -> "_AbstractAutoencoder":
        """
        Trains the autoencoder in place.

        Parameters
        ----------
        n_epochs : int
            number of epochs for training
        optimizer_params : AEOptimizerArgs
            parameters of the optimizer, includes the learning rate
        batch_size : int
            size of the data batches (default: 128)
        data : np.ndarray
            train data set. If data is passed then dataloader can remain empty (default: None)
        data_eval : np.ndarray
            evaluation data set. If data_eval is passed then evalloader can remain empty (default: None)
        dataloader : torch.utils.data.DataLoader
            dataloader to be used for training (default: default=None)
        evalloader : torch.utils.data.DataLoader
            dataloader to be used for evaluation, early stopping and learning rate scheduling if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau (default: None)
        optimizer_class : torch.optim.Optimizer
            optimizer to be used (default: torch.optim.Adam)
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction (default: torch.nn.MSELoss())
        patience : int
            patience parameter for EarlyStopping (default: 5)
        scheduler : torch.optim.lr_scheduler
            learning rate scheduler that should be used.
            If torch.optim.lr_scheduler.ReduceLROnPlateau is used then the behaviour is matched by providing the validation_loss calculated based on samples from evalloader (default: None)
        scheduler_params : dict
            dictionary of the parameters of the scheduler object (default: {})
        device : torch.device
            device to be trained on (default: torch.device('cpu'))
        model_path : str
            if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved (default: None)
        print_step : int
            specifies how often the losses are printed. If 0, no prints will occur (default: 0)
        wandb_run : Run | RunDisabled | None
            wandb run object to log the training process (default: None)
        reset_interval : int
            Intervals for reset methods.
            Parameter only has an effect if soft reset is applied during AE training.


        Returns
        -------
        self : _AbstractAutoencoder
            this instance of the autoencoder

        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        ValueError: evalloader cannot be None if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = get_dataloader(data, batch_size, True)

        if evalloader is None:
            if data_eval is not None:
                evalloader = get_dataloader(data_eval, batch_size, False)
        optimizer = setup_optimizer(
            model=self,
            dc_module=None,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            freeze_convlayers=False,
        )

        if use_scheduler := scheduler is not None:
            scheduler = scheduler(optimizer=optimizer, **scheduler_params)
            # Depending on the scheduler type we need a different step function call.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                eval_step_scheduler = True
                if evalloader is None:
                    raise ValueError(
                        "scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, but evalloader is None. Specify evalloader such that validation loss can be computed."
                    )
            else:
                eval_step_scheduler = False

        # training loop
        pbar = trange(n_epochs, desc="AE training")
        for epoch_i in pbar:
            epoch_log_dict = defaultdict(float)

            self.train()
            for batch in dataloader:
                loss_dict, _, _ = self.loss(batch, loss_fn, device)

                # Ignore warnings for torch.deterministic because they clutter the output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    optimizer.zero_grad()
                    loss_dict["loss"].backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 500, norm_type=2.0)

                # Logging
                l0_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1e10, norm_type=0)
                l1_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1e10, norm_type=1.0)
                l2_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1e10, norm_type=2.0)
                optimizer.step()

                if wandb_run is not None:
                    for k, v in loss_dict.items():
                        epoch_log_dict[f"AE train/{k}"] += v.item()

                    epoch_log_dict["AE train/l0_grad_norm"] += l0_grad_norm
                    epoch_log_dict["AE train/l1_grad_norm"] += l1_grad_norm
                    epoch_log_dict["AE train/l2_grad_norm"] += l2_grad_norm

                    if use_scheduler:
                        epoch_log_dict["AE train/lr"] = optimizer.param_groups[0]["lr"]
            # Compute epoch averages
            for k, v in epoch_log_dict.items():
                epoch_log_dict[k] = v / len(dataloader)
            epoch_log_dict["AE train/epoch"] = epoch_i

            wandb_run.log(epoch_log_dict)

            if scheduler is not None and not eval_step_scheduler:
                scheduler.step()
            # Evaluate autoencoder
            if evalloader is not None:
                # self.evaluate calls self.eval()
                val_loss = self.evaluate(dataloader=evalloader, loss_fn=loss_fn, device=device)
                if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                    print(f"Epoch {epoch_i} EVAL loss total: {val_loss.item():.6f}")

                if scheduler is not None and eval_step_scheduler:
                    scheduler.step(val_loss)

            # If there is no regularization loss key, then the defaultdict puts 0.0 in the printout
            pbar.set_postfix_str(f"Loss: {epoch_log_dict['AE train/loss']:.6f}")

        # change to eval mode after training
        self.eval()
        # Save last version of model
        if evalloader is None and model_path is not None:
            # Check if directory exists
            parent_directory = os.path.dirname(model_path)
            if parent_directory != "" and not os.path.isdir(parent_directory):
                os.makedirs(parent_directory)
            torch.save(self.state_dict(), model_path)
        # Autoencoder is now pretrained
        self.fitted = True
        return self
