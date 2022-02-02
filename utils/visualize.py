from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils


def visualize(
    imgs: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    label_names: Optional[List[str]] = None,
    grid_shape: Tuple[int, int] = (4, 4),
    figsize: Tuple[int, int] = (13, 13),
):
    if labels is None:
        return vutils.make_grid(imgs)
    if label_names is None:
        label_names = [
            "Airplane",
            "Car",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ]
    n_rows, n_cols = grid_shape
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for img, label, subplot in zip(imgs, labels, ax.ravel()):
        subplot.imshow(img.permute(1, 2, 0))
        subplot.axis("off")
        name = label_names[label]
        subplot.set_title(name)
    return fig
