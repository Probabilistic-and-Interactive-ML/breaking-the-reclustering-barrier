"""
Store type aliases for complex types.
"""

from typing import Literal

from tyro.conf import FlagConversionOff

DataSetArg = Literal[
    "cifar10",
    "mnist",
    "optdigits",
    "usps",
    "fmnist",
    "kmnist",
    "gtsrb",
    "cifar100_20",
]
""" Type for the dataset name config parameter. """

DataSubsetArg = Literal["train", "test", "all", "smallest", "train+unlabeled"]
""" Type for the dataset subset config parameter. """

ModelArg = Literal[
    "feedforward_small", "feedforward_medium", "feedforward_large", "feedforward_less_depth", "cluster_head_small"
]

""" Type for the neural network config parameter. """

ConvnetArg = Literal["resnet18", "resnet50", "none"]
""" Type for the convolutional architecture config parameter. """

ClusterArg = Literal["dec", "idec", "dcn"]
""" Type for the clustering algorithm config parameter. """

ReClusteringArg = Literal["kmeans", "kmeans++-init", "kmedoids", "em", "random"]
""" What method to use when re-clustering is enabled. """

RunnerBooleanTupleArg = FlagConversionOff[bool] | tuple[FlagConversionOff[bool], FlagConversionOff[bool]]
""" Type for boolean tuples in the runner that allows exactly two values. """

SchedulerArg = Literal["none", "step", "cosine", "linear_warmup_cosine", "linear"]
