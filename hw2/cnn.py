import torch
import torch.nn as nn
import itertools as it
from typing import Sequence
import torchvision.models as models

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: Sequence[int],
            pool_every: int,
            hidden_dims: Sequence[int],
            conv_params=dict(kernel_size=3, padding=1),
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params=dict(kernel_size=2, stride=2, padding=1)
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        for i, out_channels in enumerate(self.channels):
            conv_layer = nn.Conv2d(in_channels, out_channels, **self.conv_params)
            activation = ACTIVATIONS[self.activation_type](**self.activation_params)
            layers.append(conv_layer)
            layers.append(activation)

            if (i + 1) % self.pool_every == 0:
                pooling = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling)

            in_channels = out_channels
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self):
        """"
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.

        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            x = torch.randn(1, *self.in_size)
            features = self.feature_extractor(x)
            n_features = features.view(features.size(0), -1).shape[1]
            return n_features
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()

        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        for hidden_dim in self.hidden_dims:
            linear_layer = nn.Linear(n_features, hidden_dim)
            activation = ACTIVATIONS[self.activation_type](**self.activation_params)
            layers.append(linear_layer)
            layers.append(activation)
            n_features = hidden_dim

        output_layer = nn.Linear(n_features, self.out_classes)
        layers.append(output_layer)
        # ========================

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
            self,
            in_channels: int,
            channels: Sequence[int],
            kernel_sizes: Sequence[int],
            batchnorm: bool = False,
            dropout: float = 0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        N = len(channels)  # number of conv layers
        activation = ACTIVATIONS[activation_type]  # class names of activation function
        layers = []

        layers.append(
            nn.Conv2d(in_channels, channels[0], kernel_sizes[0], padding=int(kernel_sizes[0] / 2)))  # first layer

        for i in range(N - 1):  # the test N-1 conv layers
            if dropout > 0:  # apply dropout
                layers.append(nn.Dropout2d(dropout))
            if batchnorm:  # batchnorm
                layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(activation(**activation_params))  # activation
            # keep the size before and after conv
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_sizes[i + 1], padding=int(kernel_sizes[i + 1] / 2)))
        self.main_path = nn.Sequential(*layers)

        skip_layers = []
        if in_channels != channels[-1]:  # make the channel number the same, kernel size of 1
            skip_layers.append(nn.Conv2d(in_channels, channels[-1], 1, bias=False))
        self.shortcut_path = nn.Sequential(*skip_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
            self,
            in_out_channels: int,
            inner_channels: Sequence[int],
            inner_kernel_sizes: Sequence[int],
            **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        # ====== YOUR CODE: ======
        channels = [inner_channels[0], *inner_channels, in_out_channels]
        kernels = [1, *inner_kernel_sizes, 1]

        super().__init__(
            in_channels=in_out_channels,
            channels=channels,
            kernel_sizes=kernels,
            **kwargs
        )
        # ========================


class ResNetClassifier(ConvClassifier):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        channels = [in_channels] + self.channels
        jumps = list(range(0, len(channels) - 1, self.pool_every))
        for i in jumps:
            block_channels = channels[i + 1:i + self.pool_every + 1]
            block_kernels = [3] * len(block_channels)
            layers.append(ResidualBlock(channels[i], block_channels, block_kernels,
                                        batchnorm=self.batchnorm, dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))
            if i != jumps[-1]:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
                # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ResNetClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims):
        super().__init__(in_size,
                         out_classes,
                         channels,
                         pool_every,
                         hidden_dims,
                         dropout=0.23,  # Drop 40% of neurons
                         batchnorm=True)
