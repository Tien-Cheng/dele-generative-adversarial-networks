import torch
import torchvision.transforms as T
from torch import nn


class NormalizeInverse(T.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes):
      """Conditional Batch Normalization from conditional Spectral Normalization GAN.
      Works as a form of Batch Normalization, with the weights and biases dependent on the class.

      Implementation taken from: https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775

      :param num_features: [description]
      :type num_features: [type]
      :param num_classes: [description]
      :type num_classes: [type]
      """
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.embed = nn.Embedding(num_classes, num_features * 2)
    self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y):
    out = self.bn(x)
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out
