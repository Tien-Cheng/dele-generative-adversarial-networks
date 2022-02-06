import torch
import torch.nn as nn


class R1(nn.Module):
    """
    Implementation of the R1 GAN regularization, taken from official implementation: https://github.com/ChristophReich1996/Dirac-GAN/blob/decb8283d919640057c50ff5a1ba01b93ed86332/dirac_gan/loss.py#L292
    """

    def __init__(self, gamma):
        """
        Constructor method
        """
        # Call super constructor
        super(R1, self).__init__()
        self.gamma = gamma

    def forward(
        self, prediction_real: torch.Tensor, real_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute the regularization
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
        :param real_sample: (torch.Tensor) Batch of the corresponding real images
        :return: (torch.Tensor) Loss value
        """
        # Calc gradient
        grad_real = torch.autograd.grad(
            outputs=prediction_real.sum(),
            inputs=real_sample,
            create_graph=True,
        )[0]
        # Calc regularization
        regularization_loss: torch.Tensor = (
            self.gamma * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        )
        return regularization_loss


class HingeGANLossGenerator(nn.Module):
    """
    This class implements the Hinge generator GAN loss proposed in:
    https://arxiv.org/pdf/1705.02894.pdf
    """

    def __init__(self) -> None:
        """
        Constructor method.
        """
        # Call super constructor
        super().__init__()

    def forward(
        self, discriminator_prediction_fake: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.
        :param discriminator_prediction_fake: (torch.Tensor) Raw discriminator predictions for fake samples
        :return: (torch.Tensor) Hinge Generator GAN loss with gradient
        """
        return -discriminator_prediction_fake.mean()


class HingeGANLossDiscriminator(nn.Module):
    """
    This class implements the Hinge discriminator GAN loss proposed in:
    https://arxiv.org/pdf/1705.02894.pdf
    """

    def __init__(self) -> None:
        """
        Constructor method.
        """
        # Call super constructor
        super().__init__()

    def forward(
        self,
        discriminator_prediction_real: torch.Tensor,
        discriminator_prediction_fake: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.
        :param discriminator_prediction_real: (torch.Tensor) Raw discriminator prediction for real samples
        :param discriminator_prediction_fake: (torch.Tensor) Raw discriminator predictions for fake samples
        :return: (torch.Tensor) Hinge discriminator GAN loss
        """
        return (
            -torch.minimum(
                torch.tensor(
                    0.0, dtype=torch.float, device=discriminator_prediction_real.device
                ),
                discriminator_prediction_real - 1.0,
            ).mean()
            - torch.minimum(
                torch.tensor(
                    0.0, dtype=torch.float, device=discriminator_prediction_fake.device
                ),
                -discriminator_prediction_fake - 1.0,
            ).mean()
        )
