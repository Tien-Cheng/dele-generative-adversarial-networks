import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.nn.utils import spectral_norm


class NormalizeInverse(T.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean: tuple, std: tuple):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor: torch.Tensor):
        return super().__call__(tensor.clone())


# class ConditionalBatchNorm2d(nn.Module):
#     def __init__(self, num_features: int, num_classes: int = 10):
#         """Conditional Batch Normalization from conditional Spectral Normalization GAN.
#         Works as a form of Batch Normalization, with the weights and biases dependent on the class. Initially introduced in cGANs with Projection Discriminator.

#         Implementation taken from: https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775

#         :param num_features: [description]
#         :type num_features: [type]
#         :param num_classes: [description]
#         :type num_classes: [type]
#         """
#         super().__init__()
#         self.num_features = num_features
#         self.bn = nn.BatchNorm2d(num_features, affine=False)
#         self.embed = nn.Embedding(num_classes, num_features * 2)
#         self.embed.weight.data[:, :num_features].normal_(
#             1, 0.02
#         )  # Initialise scale at N(1, 0.02)
#         self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         out = self.bn(x)
#         gamma, beta = self.embed(y).chunk(2, 1)
#         out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
#             -1, self.num_features, 1, 1
#         )
#         return out


class ResidualBlockGenerator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        activation: callable = F.relu,
        upsample: bool = False,
        num_classes: int = 10,
        use_spectral_norm: bool = True,
    ):
        """Residual Block for the Generator, inspired by SNGANs. Implementation based on: https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py#L375

        :param in_ch: [description]
        :type in_ch: int
        :param out_ch: [description]
        :type out_ch: int
        :param kernel_size: [description], defaults to 3
        :type kernel_size: int, optional
        :param padding: [description], defaults to 1
        :type padding: int, optional
        :param activation: [description], defaults to F.relu
        :type activation: callable, optional
        :param upsample: [description], defaults to False
        :type upsample: bool, optional
        :param num_classes: [description], defaults to 10
        :type num_classes: int, optional
        :param use_spectral_norm: [description], defaults to True
        :type use_spectral_norm: bool, optional
        """
        super().__init__()
        self.activation = activation
        self.num_classes = num_classes
        self.upsample = None if upsample is False else nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.shortcut = spectral_norm(self.shortcut)
        self.bn1 = ConditionalBatchNorm2d(in_ch, num_classes)
        self.bn2 = ConditionalBatchNorm2d(out_ch, num_classes)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        h = self.bn1(x, y)
        h = self.activation(h)
        if self.upsample is not None:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.bn2(h, y)
        h = self.activation(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class ResidualBlockDiscriminator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        activation: callable = F.relu,
        downsample: bool = False,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.activation = activation
        self.downsample = None if downsample else nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.shortcut = spectral_norm(self.shortcut)

    def forward(self, x):
        h = self.activation(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample is not None:
            h = self.downsample(h)
            x = self.downsample(x)
        return h + self.shortcut(x)


class ResidualBlockDiscriminatorHead(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 128,
        kernel_size: int = 3,
        padding: int = 1,
        activation: callable = F.relu,
        use_spectral_norm: bool = True,
    ):
        """
        A residual block without batch normalization for the input to the discriminator. The reason for not having the batch normalization is because applying it to the input layer can cause instability during training.

        "Directly applying batchnorm to all layers however, resulted in sample oscillation and model instability. This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer." - DCGAN Paper



        :param in_ch: [description], defaults to 3
        :type in_ch: int, optional
        :param out_ch: [description], defaults to 128
        :type out_ch: int, optional
        :param kernel_size: [description], defaults to 3
        :type kernel_size: int, optional
        :param padding: [description], defaults to 1
        :type padding: int, optional
        :param activation: [description], defaults to F.relu
        :type activation: callable, optional
        """
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding)

        self.downsample = nn.AvgPool2d(2)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.shortcut = spectral_norm(self.shortcut)

    def forward(self, x):
        h = self.conv1(x)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.downsample(h)
        return h + self.shortcut(self.downsample(x))


class ConditionalBatchNorm2d(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        which_linear,
        eps=1e-5,
        momentum=0.1,
        norm_style="bn",
    ):
        """https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/layers.py#L278

        :param output_size: [description]
        :type output_size: [type]
        :param input_size: [description]
        :type input_size: [type]
        :param which_linear: [description]
        :type which_linear: [type]
        :param eps: [description], defaults to 1e-5
        :type eps: [type], optional
        :param momentum: [description], defaults to 0.1
        :type momentum: float, optional
        :param cross_replica: [description], defaults to False
        :type cross_replica: bool, optional
        :param mybn: [description], defaults to False
        :type mybn: bool, optional
        :param norm_style: [description], defaults to "bn"
        :type norm_style: str, optional
        """
        super().__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = spectral_norm(nn.Linear(input_size, output_size))
        self.bias = spectral_norm(nn.Linear(input_size, output_size))
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Norm style?
        self.norm_style = norm_style

        if self.norm_style in ["bn", "in"]:
            self.register_buffer("stored_mean", torch.zeros(output_size))
            self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        if self.norm_style == "bn":
            out = F.batch_norm(
                x,
                self.stored_mean,
                self.stored_var,
                None,
                None,
                self.training,
                0.1,
                self.eps,
            )
        elif self.norm_style == "in":
            out = F.instance_norm(
                x,
                self.stored_mean,
                self.stored_var,
                None,
                None,
                self.training,
                0.1,
                self.eps,
            )
        elif self.norm_style == "nonorm":
            out = x
        return out * gain + bias