import logging

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

log = logging.getLogger(__name__)


class ExampleDataModule(LightningDataModule):
    """
    Example datamodule. Replace this with your own datamodule.
    """

    def __init__(
        self,
        dataset_class: type[Dataset],
        batch_size: int,
        train_val_test_split: list[float | int],
        num_workers: int,
        pin_memory: bool,
        seed: int = 64,
    ):
        """
        Initialise the datamodule.

        :param dataset_class: The Dataset class we will wrap in this datamodule.
        **This should not be a Dataset object**, the Dataset will be instantiated during `setup()`.
        :type dataset_class: Type[Dataset]
        :param batch_size: How many samples per batch to load.
        :type batch_size: int
        :param train_val_test_split: Lengths or fractions of the training, validation and test splits.
        :type train_val_test_split: List[Union[float, int], Union[float, int], Union[float, int]]
        :param num_workers: How many subprocesses to use for data loading.
        :type num_workers: int
        :param pin_memory: If True, the dataloader will copy Tensors
            into device/CUDA pinned memory before returning them.
        :type pin_memory: bool
        :param seed: Seed for data generation.
        :type seed: int
        """

        super().__init__()
        self.save_hyperparameters()

        self.dataset_class = dataset_class
        self.batch_size = batch_size
        # This will be set in `setup`, and is equal to self.batch_size // the number of devices
        self.batch_size_per_device: int | None = None
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        # These will be set in setup
        self.data_train: Dataset | None = None
        self.data_valid: Dataset | None = None
        self.data_test: Dataset | None = None

    def setup(
        self,
        stage: str,
    ):
        """
        Perform dataset setup operations. This will be called on every device.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`. Care is taken to ensure that the data is not randomly split twice.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        :type stage: str
        """

        if stage in ["fit", "validate", "test"]:
            self.data_train, self.data_valid, self.data_test = random_split(
                self.dataset_class(),
                self.train_val_test_split,
                generator=torch.Generator().manual_seed(self.seed),
            )
            log.info(
                f"Set up datamodule with {len(self.data_train)} train samples, "
                f"{len(self.data_valid)} validation samples "
                f"and {len(self.data_test)} test samples."
            )
        elif stage == "predict":
            raise NotImplementedError("'predict' stage not yet implemented.")
        else:
            raise ValueError(f"Got an invalid stage in `setup`: {stage=}")

    def train_dataloader(self) -> DataLoader:
        if self.data_train is not None:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )
        raise ValueError("`self.data_train` is `None`. Ensure you call `self.setup` before calling this function.")

    def val_dataloader(self) -> DataLoader:
        if self.data_valid is not None:
            return DataLoader(
                dataset=self.data_valid,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        raise ValueError("`self.data_valid` is `None`. Ensure you call `self.setup` before calling this function.")

    def test_dataloader(self) -> DataLoader:
        if self.data_test is not None:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        raise ValueError("`self.data_test` is `None`. Ensure you call `self.setup` before calling this function.")
