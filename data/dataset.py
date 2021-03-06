from typing import Optional

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        transforms: Optional[list] = None,
        augments: Optional[list] = None,
    ):
        """Defines the CIFAR10 Dataset

        :param data_dir: Directory to save and load CIFAR10 dataset from
        :type data_dir: str
        :param batch_size: Number of images in each batch
        :type batch_size: int
        :param num_workers: For use in parallel processing
        :type num_workers: int
        :param transforms: Preprocessing steps to be applied to data, defaults to None (default set of transforms)
        :type transforms: Optional[list], optional
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = [T.ToTensor()] if transforms is None else transforms
        self.train_transform = (
            self.transform if augments is None else self.transform + augments
        )
        self.transform = T.Compose(self.transform)
        self.train_transform = T.Compose(self.train_transform)
        self.num_classes = 10

    def prepare_data(self):
        """Download dataset into data directory"""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Set up data splits for training, validation and testing

        :param stage: if set to "fit", only loads training and val split, if set to "test", loads test split for evaluation, defaults to None (load all splits)
        :type stage: Optional[str], optional
        """
        if stage == "fit" or stage is None:
            self.cifar_train = CIFAR10(
                self.data_dir, train=True, transform=self.train_transform
            )
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )
            self.cifar_full = ConcatDataset([self.cifar_train, self.cifar_test])
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
