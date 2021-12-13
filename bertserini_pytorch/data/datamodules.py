from re import S
import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from bertserini_pytorch.utils import constants


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str = "rsvp-ai/bertserini-bert-base-squad",
        dataset_name: str = "squad",
        dataset_task: str = None,
        task_name: str = "squad_qnli",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        """LightningDataModule to load data directly from Huggingface datasets

        Args:
            model_name_or_path (str): name to use to load pre_trained tokenizer from GLUE Transformers
            task_name (str, optional): name of the task to be trained/evaluated. Defaults to "qnli".
            max_seq_length (int, optional): [description]. Defaults to 128.
            train_batch_size (int, optional): Training batch size. Defaults to 32.
            eval_batch_size (int, optional): Validation batch size. Defaults to 32.
        """

        super().__init__()
        self.save_hyperparameters()

        self.text_fields = constants.task_text_field_map[task_name]
        self.num_labels = constants.task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, use_fast=True)

        # Rename label to labels to make it easier to pass to model forward
        # features["labels"] = example_batch["label"]

    def prepare_data(self):
        datasets.load_dataset(self.hparams.dataset_name_or_path, self.hparams.dataset_task)
        AutoTokenizer.from_pretrained(self.hparams.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams.train_batch_size)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.hparams.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.hparams.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.hparams.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.hparams.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.hparams.max_seq_length, pad_to_max_length=True, truncation=True
        )

        return features


@DATAMODULE_REGISTRY
class GLUEDataModule(BaseDataModule):

    def __init__(
        self,
        model_name: str,
        task_name: str = "qnli",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            dataset_name="glue",
            dataset_task=task_name,
            task_name=task_name,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size)

    def prepare_data(self):
        datasets.load_dataset(self.hparams.dataset_name_or_path, self.hparams.task_name)
        AutoTokenizer.from_pretrained(self.hparams.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        """This class is called to prepare the datasets before starting to load the data

        Args:
            stage (str): string representing which stage is about to strart (fit/test/eval/...)
        """

        self.dataset = datasets.load_dataset(self.hparams.dataset_name, self.hparams.dataset_task)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in constants.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]


@DATAMODULE_REGISTRY
class SQuADDataModule(BaseDataModule):
    def __init__(
        self,
        model_name_or_path: str = "rsvp-ai/bertserini-bert-base-squad",
        task_name: str = "squad",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
    ):
        super().__init__(
            model_path=model_name_or_path,
            dataset_name_or_path="squad",
            dataset_task=None,
            task_name=task_name,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size)

    def setup(self, stage: str):
        """This class is called to prepare the datasets before starting to load the data

        Args:
            stage (str): string representing which stage is about to strart (fit/test/eval/...)
        """

        self.dataset = datasets.load_dataset(self.hparams.dataset_name, self.hparams.dataset_task)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.columns = [c for c in self.dataset[split].column_names if c in constants.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
