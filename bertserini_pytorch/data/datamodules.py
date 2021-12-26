import os
from re import S
from typing import List
import datasets
from datasets.utils.file_utils import estimate_dataset_size
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import BertTokenizer

from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from transformers.data.processors.squad import SquadV2Processor
from bertserini_pytorch.utils import constants
from transformers import squad_convert_examples_to_features
import requests
from bertserini_pytorch.utils.base import Context, Question
from datasets import load_dataset

from bertserini_pytorch.utils.pyserini import craft_squad_examples


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

    # fit,validate,test,predict
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
class PredictionDataModule(LightningDataModule):
    def __init__(
        self,
        question: Question,
        contexts: List[Context],
        model_name: str
    ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.question = question
        self.contexts = contexts

    def setup(self, stage=None):
        # all examples, features and dataset are stored because
        # they are needed later to compute aggregated features

        # convert question and contexts to list of SquadExamples objects
        self.examples = craft_squad_examples(self.question, self.contexts)

        # convert list of SquadExamples objects to features
        self.features, self.dataset = squad_convert_examples_to_features(
            examples=self.examples,
            tokenizer=self.tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=1,
            tqdm_enabled=True
        )

    def predict_dataloader(self):
        sampler = SequentialSampler(self.dataset)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=1)

        return dataloader


@DATAMODULE_REGISTRY
class SQuADDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str = "squad_v2",
        model_path: str = "rsvp-ai/bertserini-bert-base-squad",
        train_batch_size: int = 32,
        val_batch_size: int = 1,
        doc_stride: int = 128,
        max_query_length: int = 64,
        max_seq_length: int = 387,
        threads: int = 6,
        workers: int = 12,
    ):
        """PytorchLightning DataModule implementation for squad_v2 dataset. The dataloaders output ready-to-train tensors.

        Args:
            dataset_path (str, optional): [description]. Defaults to "squad_v2".
            model_path (str, optional): [description]. Defaults to "rsvp-ai/bertserini-bert-base-squad".
            train_batch_size (int, optional): [description]. Defaults to 32.
            val_batch_size (int, optional): [description]. Defaults to 32.
            max_seq_length (int, optional): [description]. Defaults to 387.
        """
        super().__init__()

        # avoids a huge list of self.var_name = var_name
        self.save_hyperparameters()
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_path)

    def prepare_data(self):

        # Download the dev version of squad if does not exists already
        if not os.path.isfile("./tmp/squad/dev-v2.0.json"):
            dataset = requests.get(
                url='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json', allow_redirects=True)
            os.makedirs('./tmp/squad', exist_ok=True)
            with open('./tmp/squad/dev-v2.0.json', 'wb') as fb:
                fb.write(dataset.content)

        self.processor = SquadV2Processor()

    # fit,validate,test,predict

    def setup(self, stage: str):
        """This class is called to prepare the datasets before starting to load the data

        Args:
            stage (str): string representing which stage is about to strart (fit/test/eval/...)
        """
        if stage is None or stage == "fit":
            examples = self.processor.get_train_examples("./tmp/squad/")
        elif stage == "validate":
            examples = self.processor.get_dev_examples("./tmp/squad/")

        # Convert the dataset of SQuAD objects to features ready to be fed to a model
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.hparams.max_seq_length,
            doc_stride=self.hparams.doc_stride,
            max_query_length=self.hparams.max_query_length,
            is_training=stage == 'fit',
            return_dataset="pt",
            threads=self.hparams.threads,
        )

        # these references are needed to compute statistics over the dataset
        self.dataset = dataset
        self.examples = examples
        self.features = features

    # TODO: Implement other dataloaders

    def val_dataloader(self):
        eval_sampler = SequentialSampler(self.dataset)
        eval_dataloader = DataLoader(self.dataset, sampler=eval_sampler,
                                     batch_size=self.hparams.val_batch_size, num_workers=self.hparams.workers)
        return eval_dataloader
