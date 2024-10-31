import torch

from config.base_config import Args
from src.deep.autoencoders import ConvolutionalAutoencoder, FeedforwardAutoencoder


def determine_projector_layers(args):
    if args.projector_depth > 0:
        if args.projector_layer_size is None:
            layer_size = args.embedding_dim
        else:
            layer_size = args.projector_layer_size
        # embedding dim is the same for input and output
        layers = [args.embedding_dim] + [layer_size] * args.projector_depth + [args.embedding_dim]
    else:
        layers = None
    return layers


def determine_first_resnet_fc_layer_dim(resnet_type: str) -> int:
    if resnet_type == "resnet18":
        dim = 512
    elif resnet_type == "resnet50":
        dim = 2048
    else:
        raise ValueError(f"Invalid convnet encoder type {resnet_type} specified.")
    return dim


def initialize_autoencoder(args: Args, data: torch.Tensor, device: torch.device) -> FeedforwardAutoencoder:
    """Initializes an autoencoder model based on the specifications in args."""

    if args.model_type == "feedforward_small":
        # Smaller network
        layers = [data.shape[-1], 256, 128, 64, args.embedding_dim]
    elif args.model_type == "feedforward_medium":
        # Medium network
        layers = [data.shape[-1], 512, 256, args.embedding_dim]
    elif args.model_type == "feedforward_large":
        # Bigger network
        layers = [data.shape[-1], 1024, 512, 256, args.embedding_dim]

    elif args.model_type == "feedforward_less_depth":
        layers = [data.shape[-1], 512, args.embedding_dim]

    elif args.model_type == "cluster_head_small":
        layers = [data.shape[-1], 256, 128, args.embedding_dim]

    else:
        raise ValueError(f"Invalid model type {args.model_type} specified.")

    if args.activation_fn == "leaky_relu":
        activation_fn = torch.nn.LeakyReLU
    elif args.activation_fn == "relu":
        activation_fn = torch.nn.ReLU
    else:
        raise ValueError(f"Invalid activation fn type {args.activation_fn} specified.")

    # determine projector size for contrastive learning
    if args.use_contrastive_loss:
        projector_layers = determine_projector_layers(args)
    else:
        projector_layers = None

    if args.convnet == "none":
        ae = FeedforwardAutoencoder(
            layers=layers,
            projector_layers=projector_layers,
            activation_fn=activation_fn,
            batch_norm=args.batch_norm,
            use_contrastive_loss=args.use_contrastive_loss,
            contrastive_loss_params={"tau": args.softmax_temperature_tau},
            normalize_embeddings=args.normalize_embeddings,
        ).to(device)
    elif args.convnet == "resnet18" or args.convnet == "resnet50":
        # First dimension of layers for fully connected layers need to be changed to
        # last output of resnet after AverageAdaptivePooling
        first_dim = determine_first_resnet_fc_layer_dim(args.convnet)
        layers[0] = first_dim
        if projector_layers is not None and args.separate_cluster_head:
            # Replace first and last dim with output of resnet
            projector_layers[0] = first_dim
            projector_layers[-1] = first_dim
        ae = ConvolutionalAutoencoder(
            input_height=data.shape[-1],
            fc_layers=layers,
            projector_layers=projector_layers,
            conv_encoder_name=args.convnet,
            activation_fn=activation_fn,
            use_contrastive_loss=args.use_contrastive_loss,
            contrastive_loss_params={"tau": args.softmax_temperature_tau},
            separate_cluster_head=args.separate_cluster_head,
            fc_kwargs={"batch_norm": args.batch_norm, "additional_last_linear_layer": args.additional_last_linear_layer},
            normalize_embeddings=args.normalize_embeddings,
        ).to(device)
    else:
        raise ValueError(f"Invalid convnet encoder type {args.convnet} specified.")

    return ae
