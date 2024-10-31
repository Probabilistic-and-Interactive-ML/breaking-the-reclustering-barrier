import warnings

import torch
from torchvision.models._api import Weights

from ._abstract_autoencoder import FullyConnectedBlock, _AbstractAutoencoder
from ._resnet_ae_modules import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
from .simclr_loss import SimclrLoss

_VALID_CONV_MODULES = {
    "resnet18": {
        "enc": resnet18_encoder,
        "dec": resnet18_decoder,
    },
    "resnet50": {
        "enc": resnet50_encoder,
        "dec": resnet50_decoder,
    },
}

_CONV_MODULES_INPUT_DIM = {"resnet18": 512, "resnet50": 2048}


class ConvolutionalAutoencoder(_AbstractAutoencoder):
    """
    A convolutional autoencoder based on the ResNet architecture.

    Parameters
    ----------
    input_height: int
        height of the images for the decoder (assume square images)
    fc_layers : list
        list of the different layer sizes from flattened convolutional layer input to embedding. First input needs to be 512 if conv_encoder_name="resnet18" and 2048 if conv_encoder_name="resnet50".
        If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
    projector_layers: list | None
        layers of projector for contrastive learning put on on top of embedding. If None, no projection head will be used
        e.g. SimCLR used projector_layers=[embedding_dim, embedding_dim, output_dim] (default: None)
    conv_encoder_name : str
        name of convolutional resnet encoder part of the autoencoder. Can be 'resnet18' or 'resnet50' (default: 'resnet18')
    conv_decoder_name : str
        name of convolutional resnet decoder part of the autoencoder. Can be 'resnet18' or 'resnet50'. If None it will be the same as conv_encoder_name (default: None)
    activation_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the hidden layers, if None then it will be linear (default: torch.nn.LeakyReLU)
    fc_decoder_layers : list
        list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers (default: None)
    decoder_output_fn : torch.nn.Module
        activation function from torch.nn, set the activation function for the decoder output layer, if None then it will be linear.
        E.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1 (default: None)
    pretrained_encoder_weights : torchvision.models._api.Weights
        weights from torch.vision.models, indicates whether pretrained resnet weights should be used for the encoder. (default: None)
    pretrained_decoder_weights : torchvision.models._api.Weights
        weights from torch.vision.models, indicates whether pretrained resnet weights should be used for the decoder (not implemented yet). (default: None)
    reusable : bool
        If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
        Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
        As copies of this object are created, the memory requirement increases (default: True)
    fc_kwargs : dict
        additional parameters for FullyConnectedBlock, e.g. batch_norm or dropout
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
    separate_cluster_head: bool (default: False)
        If set to True, projector will be added to output of resnet instead of the output of the MLP cluster head.
        NOTE: Works currently only for Resnet architecture

    Attributes
    ----------
    conv_encoder : ResNetEncoder
        convolutional resnet encoder part of the autoencoder
    conv_decoder : ResNetEncoder
        convolutional resnet decoder part of the autoencoder
    fc_encoder : FullyConnectedBlock
        fully connected encoder part of the convolutional autoencoder, together with conv_encoder is responsible for embedding data points
    fc_decoder : FullyConnectedBlock
        fully connected decoder part of the convolutional autoencoder, together with conv_decoder is responsible for reconstructing data points from the embedding
    fitted  : bool
        indicates whether the autoencoder is already fitted
    reusable : bool
        indicates whether the autoencoder should be reused by multiple deep clustering algorithms

    References
    ----------
    He, Kaiming, et al. "Deep residual learning for image recognition."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

    and

    LeCun, Yann, et al. "Backpropagation applied to handwritten zip code recognition."
    Neural computation 1.4 (1989): 541-551.
    """

    def __init__(
        self,
        input_height: int,
        fc_layers: list,
        projector_layers: list = None,
        conv_encoder_name: str = "resnet18",
        conv_decoder_name: str = None,
        activation_fn: torch.nn.Module = torch.nn.ReLU,
        fc_decoder_layers: list = None,
        decoder_output_fn: torch.nn.Module = None,
        pretrained_encoder_weights: Weights = None,
        pretrained_decoder_weights: Weights = None,
        reusable: bool = True,
        use_contrastive_loss: bool = False,
        contrastive_loss_fn: torch.nn.Module = SimclrLoss,
        contrastive_loss_params: dict = {
            "tau": 0.1,
        },
        fc_kwargs: dict = {},
        separate_cluster_head: bool = False,
        normalize_embeddings: bool = False,
    ):
        super().__init__(
            reusable=reusable,
            use_contrastive_loss=use_contrastive_loss,
            separate_cluster_head=separate_cluster_head,
        )
        self.input_height = input_height
        # Check if layers match
        if fc_decoder_layers is None:
            fc_decoder_layers = fc_layers[::-1]
        if fc_layers[-1] != fc_decoder_layers[0]:
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {fc_layers[-1]} and {fc_decoder_layers[0]} respectively."
            )

        self.contrastive_loss_fn = contrastive_loss_fn(**contrastive_loss_params)

        if projector_layers is None and use_contrastive_loss:
            warnings.warn("projector_layers are None -> You are using contrastive loss without projector.")
        if projector_layers is not None and not use_contrastive_loss:
            raise ValueError("projector_layers are specified but use_contrastive loss is set to False.")
        # Initialize projector for Contrastive Learning
        self.projector_layers = projector_layers
        if self.projector_layers is not None:
            _layer_to_check = fc_layers[0] if self.separate_cluster_head else fc_layers[-1]
            if _layer_to_check != self.projector_layers[0]:
                raise ValueError(
                    f"Hidden layer and first projector layer do not match, they are {_layer_to_check} and {self.projector_layers[0]} respectively."
                )
            # simclr used: torch.nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), torch.nn.Sequential(nn.Linear(dim_mlp, output_dim))
            self.projector = FullyConnectedBlock(
                layers=self.projector_layers,
                batch_norm=None,
                dropout=None,
                activation_fn=activation_fn,
                bias=True,
                output_fn=None,
            )
        else:
            self.projector = torch.nn.Identity()

        # Setup convolutional encoder and decoder
        if conv_decoder_name is None:
            conv_decoder_name = conv_encoder_name
        if conv_encoder_name in _VALID_CONV_MODULES:
            if fc_layers[0] != _CONV_MODULES_INPUT_DIM[conv_encoder_name]:
                raise ValueError(
                    f"First input in fc_layers needs to be {_CONV_MODULES_INPUT_DIM[conv_encoder_name]} for {conv_encoder_name} architecture, but is fc_layers[0] = {fc_layers[0]}"
                )
            self.conv_encoder = _VALID_CONV_MODULES[conv_encoder_name]["enc"](
                pretrained_weights=pretrained_encoder_weights,
                arch=conv_encoder_name,
            )
        else:
            raise ValueError(
                f"value for conv_encoder_name={conv_encoder_name} is not valid. Has to be one of {list(_VALID_CONV_MODULES.keys())}"
            )
        if conv_decoder_name in _VALID_CONV_MODULES:
            self.conv_decoder = _VALID_CONV_MODULES[conv_decoder_name]["dec"](
                latent_dim=fc_decoder_layers[-1],
                input_height=self.input_height,
                pretrained_weights=pretrained_decoder_weights,
            )
        else:
            raise ValueError(
                f"value for conv_decoder_name={conv_decoder_name} is not valid. Has to be one of {list(_VALID_CONV_MODULES.keys())}"
            )

        # Initialize encoder
        self.fc_encoder = FullyConnectedBlock(layers=fc_layers, activation_fn=activation_fn, output_fn=None, **fc_kwargs)
        # Inverts the list of layers to make symmetric version of the encoder
        fc_kwargs["additional_last_linear_layer"] = False
        self.fc_decoder = FullyConnectedBlock(
            layers=fc_decoder_layers, activation_fn=activation_fn, output_fn=decoder_output_fn, **fc_kwargs
        )
        self.normalize_embeddings = normalize_embeddings

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the encoder function to x. Runs x through the conv_encoder and then the fc_encoder.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points

        Returns
        -------
        embedded : torch.Tensor
            the embedded data point with dimensionality embedding_size
        """
        embedded = self.conv_encoder(x)
        embedded = self.fc_encoder(embedded)
        if self.normalize_embeddings:
            embedded = torch.nn.functional.normalize(embedded, dim=1)
        return embedded

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply the decoder function to embedded. Runs x through the conv_decoder and then the fc_decoder.

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        decoded : torch.Tensor
            returns the reconstruction of embedded
        """
        decoded = self.fc_decoder(embedded)
        decoded = self.conv_decoder(decoded)
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
