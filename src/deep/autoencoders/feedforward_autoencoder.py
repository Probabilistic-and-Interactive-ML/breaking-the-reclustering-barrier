import warnings

import torch

from ._abstract_autoencoder import FullyConnectedBlock, _AbstractAutoencoder
from .simclr_loss import SimclrLoss


class FeedforwardAutoencoder(_AbstractAutoencoder):
    """
    A flexible feedforward autoencoder.

    Parameters
    ----------
    layers : list
        list of the different layer sizes from input to embedding, e.g. an example architecture for MNIST [784, 512, 256, 10], where 784 is the input dimension and 10 the embedding dimension.
        If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    projector_layers: list | None
        layers of projector for contrastive learning put on on top of embedding. If None, no projection head will be used
        e.g. SimCLR used projector_layers=[embedding_dim, embedding_dim, output_dim] (default: None)
    batch_norm : bool
        Set True if you want to use torch.nn.BatchNorm1d (default: False)
    dropout : float
        Set the amount of dropout you want to use (default: None)
    activation_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: torch.nn.LeakyReLU)
    bias : bool
        set False if you do not want to use a bias term in the linear layers (default: True)
    decoder_layers : list
        list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers (default: None)
    decoder_output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the decoder output layer, if None then it will be linear.
        E.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1 (default: None)
    reusable : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)
    use_contrastive_loss : bool
        If set to true, contrastive loss will be used instead of reconstruction loss
    contrastive_loss_fn: torch.nn.Module
        Contrastive loss to be used (default: SimclrLoss)
    contrastive_loss_params: dict
        hyperparameters for contrastive_loss=SimclrLoss (default:
        dict = {
            # Temperature parameter of Softmax
            "tau": 0.1,
        })

    Attributes
    ----------
    encoder : FullyConnectedBlock
        encoder part of the autoencoder, responsible for embedding data points (class is FullyConnectedBlock)
    decoder : FullyConnectedBlock
        decoder part of the autoencoder, responsible for reconstructing data points from the embedding (class is FullyConnectedBlock)
    fitted  : bool
        indicates whether the autoencoder is already fitted
    reusable : bool
        indicates whether the autoencoder should be reused by multiple deep clustering algorithms

    References
    ----------
    E.g. Ballard, Dana H. "Modular learning in neural networks." Aaai. Vol. 647. 1987.
    """

    def __init__(
        self,
        layers: list,
        projector_layers: list = None,
        batch_norm: bool = False,
        dropout: float = None,
        activation_fn: torch.nn.Module = torch.nn.LeakyReLU,
        bias: bool = True,
        decoder_layers: list = None,
        decoder_output_fn: torch.nn.Module = None,
        reusable: bool = True,
        use_contrastive_loss: bool = False,
        contrastive_loss_fn: torch.nn.Module = SimclrLoss,
        contrastive_loss_params: dict = {
            "tau": 0.1,
        },
        additional_last_linear_layer: bool = False,
        normalize_embeddings: bool = False,
    ):
        super().__init__(
            reusable=reusable,
            use_contrastive_loss=use_contrastive_loss,
        )
        if decoder_layers is None:
            decoder_layers = layers[::-1]
        if layers[-1] != decoder_layers[0]:
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {layers[-1]} and {decoder_layers[0]} respectively."
            )
        if layers[0] != decoder_layers[-1]:
            raise ValueError(
                f"Output and input dimension do not match, they are {layers[0]} and {decoder_layers[-1]} respectively."
            )
        # Initialize encoder
        self.encoder = FullyConnectedBlock(
            layers=layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=activation_fn,
            bias=bias,
            output_fn=None,
            additional_last_linear_layer=additional_last_linear_layer,
        )

        self.contrastive_loss_fn = contrastive_loss_fn(**contrastive_loss_params)

        if projector_layers is None and use_contrastive_loss:
            warnings.warn("projector_layers are None -> You are using contrastive loss without projector.")
        if projector_layers is not None and not use_contrastive_loss:
            raise ValueError("projector_layers are specified but use_contrastive loss is set to False.")
        # Initialize projector for Contrastive Learning
        self.projector_layers = projector_layers
        if self.projector_layers is not None:
            if layers[-1] != self.projector_layers[0]:
                raise ValueError(
                    f"Innermost hidden layer and first projector layer do not match, they are {layers[-1]} and {self.projector_layers[0]} respectively."
                )
            # simclr used: torch.nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), torch.nn.Sequential(nn.Linear(dim_mlp, output_dim))
            self.projector = FullyConnectedBlock(
                layers=self.projector_layers,
                batch_norm=None,
                dropout=None,
                activation_fn=activation_fn,
                bias=bias,
                output_fn=None,
            )
        else:
            self.projector = torch.nn.Identity()

        # Inverts the list of layers to make symmetric version of the encoder
        self.decoder = FullyConnectedBlock(
            layers=decoder_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=activation_fn,
            bias=bias,
            output_fn=decoder_output_fn,
        )
        self.normalize_embeddings = normalize_embeddings

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the encoder function to x.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points

        Returns
        -------
        embedded : torch.Tensor
            the embedded data point with dimensionality embedding_size
        """
        assert (
            x.shape[1] == self.encoder.layers[0]
        ), f"Input layer of the encoder ({self.encoder.layers[0]}) does not match input sample ({x.shape[1]})"
        embedded = self.encoder(x)
        if self.normalize_embeddings:
            embedded = torch.nn.functional.normalize(embedded, dim=1)
        return embedded

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply the decoder function to embedded.

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        decoded : torch.Tensor
            returns the reconstruction of embedded
        """
        assert embedded.shape[1] == self.decoder.layers[0], "Input layer of the decoder does not match input sample"
        decoded = self.decoder(embedded)
        return decoded

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply the projector function to encoded z.

        Parameters
        ----------
        z : torch.Tensor
            encoded input data point, can also be a mini-batch of points

        Returns
        -------
        projected : torch.Tensor
            the projected data point with dimensionality self.projector_layers[-1]
        """
        if self.projector_layers is None:
            projected = z
        else:
            assert (
                z.shape[1] == self.projector.layers[0]
            ), f"Input layer of the projector ({self.projector.layers[0]}) does not match the encoded input sample ({z.shape[1]})"
            projected = self.projector(z)
        return projected
