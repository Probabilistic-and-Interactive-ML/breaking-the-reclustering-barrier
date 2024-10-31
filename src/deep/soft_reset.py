import math

import torch
import torch.nn as nn


@torch.no_grad()
def _batchnorm_reset(layer: nn.BatchNorm1d | nn.BatchNorm2d, reset_interpolation_factor: float) -> None:
    """Reset the weights and the running mean and variance of a batchnorm layer."""

    # Sample phi from initializer distribution
    phi_weight = torch.ones(layer.weight.data.shape).to(layer.weight.device)

    # Apply soft reset to weights
    layer.weight.data.mul_(reset_interpolation_factor)
    layer.weight.data.add_((1 - reset_interpolation_factor) * phi_weight)
    # Bias init is zero, so no need to add phi_bias
    layer.bias.data.mul_(reset_interpolation_factor)


@torch.no_grad()
def _conv_linear_kaiming_reset(layer: nn.Linear | nn.Conv2d, reset_interpolation_factor: float) -> None:
    """
    Reset weights by applying a soft reset.
    θ_t = alpha*θ_t-1 + (1-alpha)*phi
    """
    # Sample phi from initializer distribution
    phi = torch.empty(layer.weight.data.shape).to(layer.weight.device)
    nn.init.kaiming_uniform_(phi)
    # Apply transformation
    layer.weight.data.mul_(reset_interpolation_factor)
    layer.weight.data.add_((1 - reset_interpolation_factor) * phi)

    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight.data)
        phi_bias = torch.empty(layer.bias.data.shape).to(layer.bias.device)
        if isinstance(layer, nn.Conv2d):
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(phi_bias, -bound, bound)
                layer.bias.data.mul_(reset_interpolation_factor)
                layer.bias.data.add_((1 - reset_interpolation_factor) * phi_bias)
        else:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(phi_bias, -bound, bound)
            layer.bias.data.mul_(reset_interpolation_factor)
            layer.bias.data.add_((1 - reset_interpolation_factor) * phi_bias)


def _check_layer_reset_eligibility(
    layer: nn.Module, embedding_layer: nn.Linear, reset_embedding: bool, reset_batchnorm: bool
) -> bool:
    """Check if a layer is eligible for soft reset."""
    if isinstance(layer, nn.Linear | nn.Conv2d | nn.BatchNorm1d | nn.BatchNorm2d) and not layer.weight.requires_grad:
        return False
    elif reset_embedding and isinstance(layer, nn.Linear | nn.Conv2d):
        # Reset all layers irrespective of whether they are the embedding layer or not
        return True
    elif not reset_embedding and isinstance(layer, nn.Conv2d):
        # Convolutional layers are always reset because they can't be the embedding layer
        return True
    elif not reset_embedding and isinstance(layer, nn.Linear):
        # Linear layers are reset only if they are not the embedding layer
        return layer is not embedding_layer
    elif reset_batchnorm and isinstance(layer, nn.BatchNorm1d | nn.BatchNorm2d):
        return True
    else:
        return False


@torch.no_grad()
def soft_reset(
    autoencoder: nn.Module,
    reset_interpolation_factor: float,
    reset_interpolation_factor_step: float,
    reset_batchnorm: bool,
    reset_embedding: bool,
    reset_projector: bool,
    reset_convlayers: bool,
) -> None:
    """
    Reset weights of each linear submodule by applying a soft reset.
    θ_t = alpha*θ_t-1 + (1-alpha)*phi
    """

    candidate_layers_dict = {}
    reset_factors_dict = {}
    # reset strength of fully connected encoder
    reset_factors_dict["fc_modules"] = reset_interpolation_factor
    # isinstance check is not possible as ConvolutionalAutoencoder cannot be imported,
    # due to circular import with _abstract_autoencoder.py and shrink_perturb.py
    if autoencoder._get_name() == "ConvolutionalAutoencoder":
        # extract embedding layer
        embedding_layer = autoencoder.fc_encoder.block[-1]
        candidate_layers_dict["fc_modules"] = list(autoencoder.fc_encoder.modules())
        # tyro parses the reset_convlayers argument as string thus we need to check for "True" and "False" values via strings
        # NOTE: We are not sure if this a bug of tyro or intended behavior, but as we allow boolean values in the config as well we have this additional check for truthness
        if (
            isinstance(reset_convlayers, str)
            and reset_convlayers == "True"
            or isinstance(reset_convlayers, bool)
            and reset_convlayers
        ):
            # RESET ALL CONVLAYERS
            #
            # in this case all convlayers are reset with the same reset_interpolation_factor as the fc_encoder
            candidate_layers_dict["conv_layers"] = list(autoencoder.conv_encoder.modules())
            # NOTE: We only explore the contrastive setting without decoder for the convolutional network, so conv_decoder layers are not reset
            # candidate_layers_dict["conv_layers"] += list(autoencoder.conv_decoder.modules())
            reset_factors_dict["conv_layers"] = reset_interpolation_factor
        elif isinstance(reset_convlayers, str) and reset_convlayers != "False":
            # RESET USER SPECIFIED CONVLAYERS
            #
            # split string such as "2,3,4" into list ["2","3","4"]
            convlayers_to_be_reset = reset_convlayers.split(",")
            # convert list of strings to list of ints and sort descendingly to make sure we use always the same order
            # and skip empty string
            convlayers_to_be_reset = [int(i.strip()) for i in convlayers_to_be_reset if i.strip() != ""]
            convlayers_to_be_reset.sort(reverse=True)
            # loop over convlayers from deepest to earliest layer
            # deeper layer are reset more strongly
            # e.g. for resnet18, convlayer 4 is the one before the pooling operation
            for idx, i in enumerate(convlayers_to_be_reset):
                layer_i = getattr(autoencoder.conv_encoder, f"layer{i}")
                resnet_layer_name = f"layer{i}"
                candidate_layers_dict[resnet_layer_name] = list(layer_i.modules())
                reset_factor_i = reset_interpolation_factor + (idx + 1) * reset_interpolation_factor_step
                if reset_factor_i > 1:
                    # reset factor is capped at one, meaning no reset will happen
                    reset_factors_dict[resnet_layer_name] = 1.0
                    print(f"Warning: convlayer {resnet_layer_name} will not be reset as reset factor is 1.0")
                else:
                    reset_factors_dict[resnet_layer_name] = reset_factor_i
    else:
        # NO CONVLAYER RESET
        # extract embedding layer
        embedding_layer = autoencoder.encoder.block[-1]
        candidate_layers_dict["fc_modules"] = list(autoencoder.encoder.modules())

    if reset_projector:
        candidate_layers_dict["projector"] = list(autoencoder.projector.modules())
        reset_factors_dict["projector"] = reset_interpolation_factor

    # Filter the layers that are eligible for soft reset
    for key, candidate_layers in candidate_layers_dict.items():
        # Filter the layers that are eligible for soft reset
        candidate_layers_dict[key] = (
            layer
            for layer in candidate_layers
            if _check_layer_reset_eligibility(layer, embedding_layer, reset_embedding, reset_batchnorm)
        )
    # Reset the eligible layers
    print(f"Applying Soft Reset with alpha={reset_interpolation_factor}.")
    for key, layers_to_be_reset in candidate_layers_dict.items():
        reset_factor_i = reset_factors_dict[key]
        for layer in layers_to_be_reset:
            if isinstance(layer, nn.Conv2d):
                # Assume order of convlayer_reset_interpolation_factors is the same as number of convlayers to be reset
                print(f"reset convlayer {key} with alpha={reset_factor_i}")
                _conv_linear_kaiming_reset(layer, reset_factor_i)
            elif isinstance(layer, nn.Linear):
                print(f"reset linear layer {key} with alpha={reset_factor_i}")
                _conv_linear_kaiming_reset(layer, reset_interpolation_factor)
            elif isinstance(layer, nn.BatchNorm1d | nn.BatchNorm2d):
                _batchnorm_reset(layer, reset_interpolation_factor)
                print(f"reset batchnorm layer {key} with alpha={reset_factor_i}")
            else:
                raise ValueError(f"Layer {layer} is not supported for soft reset.")
