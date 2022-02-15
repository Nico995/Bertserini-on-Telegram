from collections import defaultdict
import os
from re import S
from typing import List
import datasets
from datasets.utils.file_utils import estimate_dataset_size
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import BertTokenizer
from bertserini_on_telegram.utils.pyserini import Searcher

from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from transformers.data.processors.squad import SquadV2Processor, SquadV1Processor
from bertserini_on_telegram.utils import constants
from transformers import squad_convert_examples_to_features
import requests
from bertserini_on_telegram.utils.base import Context, Question
from datasets import load_dataset

from bertserini_on_telegram.utils.pyserini import craft_squad_examples
from transformers.data.metrics.squad_metrics import normalize_answer


@DATAMODULE_REGISTRY
class PredictionDataModule(LightningDataModule):
    """A DataModule standardizes the training, val, test splits, data preparation and transforms. 
    The main advantage is consistent data splits, data preparation and transforms across models.
    This allows you to share a full dataset without explaining how to download, split, transform, and process the data.

    This particular DataModule is used at inference time, when we need to predict based on multiple question-context pairs
    of type (q1, c1), ..., (q1, cn), where c1, ..., c2 come from a pyserini search.

    Args:
        question (Question): The Question that the user asked.
        contexts (List[Context]): The list of retreived Context objects to use as contexts for the answer.
        model_name (str): The name of the model used for inference. Needed to load the correct tokenizer.
    """

    def __init__(
        self,
        question: Question,
        contexts: List[Context],
        model_name: str,
    ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.question = question
        self.contexts = contexts

        self.pyserini_scores = []
        for context in self.contexts:
            self.pyserini_scores.append(context.score)

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
        """
        Returns the dataloader object to load samples
        """
        sampler = SequentialSampler(self.dataset)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=1)

        return dataloader


@DATAMODULE_REGISTRY
class SQuADDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str = "rsvp-ai/bertserini-bert-base-squad",
        train_batch_size: int = 32,
        val_batch_size: int = 1,
        doc_stride: int = 128,
        max_query_length: int = 64,
        max_seq_length: int = 387,
        threads: int = 6,
        workers: int = 12,
        num_contexts: int = 3
    ):
        """A DataModule standardizes the training, val, test splits, data preparation and transforms. 
        The main advantage is consistent data splits, data preparation and transforms across models.
        This allows you to share a full dataset without explaining how to download, split, transform, and process the data.

        PytorchLightning DataModule implementation for squad dataset. The dataloaders output ready-to-train tensors.

        Args:
            model_name (str, optional): The name of the pretrained model to use. Defaults to "rsvp-ai/bertserini-bert-base-squad".
            train_batch_size (int, optional): Batch size for the training dataloader. Defaults to 32.
            val_batch_size (int, optional): Batch size for the validation dataloader. Defaults to 1.
            doc_stride (int, optional): The stride used when the context is too large and is split across several features. Defaults to 128.
            max_query_length (int, optional): The maximum length of the query. Defaults to 64.
            max_seq_length (int, optional): The maximum sequence length of the inputs. Defaults to 387.
            threads (int, optional): Number of threads to deploy when converting examples to features. Defaults to 6.
            workers (int, optional): Number of processes to deploy for data loading. Defaults to 12.      
            num_contexts (int, optional): Number of contexts that PySerini needs to retrieve
        """
        super().__init__()

        # avoids a huge list of self.var_name = var_name
        self.save_hyperparameters()
        self.searcher = Searcher("enwiki-paragraphs")
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_name)

    def prepare_data(self):
        """Called first. This method handels downloading data from the internet if necessary.

        """
        # Download the dev version of squadv1 if does not exists already and write it to disk
        if not os.path.isfile("./tmp/squad/dev-v1.1.json"):
            dataset = requests.get(
                url='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json', allow_redirects=True)
            os.makedirs('./tmp/squad', exist_ok=True)
            with open('./tmp/squad/dev-v1.1.json', 'wb') as fb:
                fb.write(dataset.content)

        # Download the train version of squad if does not exists already and write it to disk
        if not os.path.isfile("./tmp/squad/train-v1.1.json"):
            dataset = requests.get(
                url='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json', allow_redirects=True)
            os.makedirs('./tmp/squad', exist_ok=True)
            with open('./tmp/squad/train-v1.1.json', 'wb') as fb:
                fb.write(dataset.content)

        self.processor = SquadV1Processor()

    def setup(self, stage: str):
        """This class is called to prepare the datasets before starting to load the data.

        Args:
            stage (str): string representing which stage is about to strart (fit/test/eval/...)

        """

        # We currently limit
        if stage is None or stage == "fit":
            examples = self.processor.get_dev_examples("./tmp/squad/", filename='train-v1.1.json')[:1000]
        elif stage == "validate":
            examples = self.processor.get_dev_examples("./tmp/squad/", filename='dev-v1.1.json')[:1000]

        # delete the original squad context and insert 1 entry for each bertserini retrieved context
        new_examples = []
        pyserini_scores = defaultdict(lambda: [])
        # Keep track of pyserini scores for all contexts
        for i, example in enumerate(examples):
            question = Question(example.question_text, "en")
            contexts = self.searcher.retrieve(question, self.hparams.num_contexts)
            contexts_examples = craft_squad_examples(question, contexts)

            ex_answer_text = [answer['text'] for answer in example.answers]
            ex_answers = example.answers

            for j, ctx_ex in enumerate(contexts_examples):

                ctx_ex.has_answer = 0

                # check whether the answer exists in any of the contexts
                for answer_text in ex_answer_text:
                    if normalize_answer(answer_text) in normalize_answer(ctx_ex.context_text):
                        ctx_ex.has_answer = 1

                # Copy the old information into the new examples
                ctx_ex.answers = ex_answers
                ctx_ex.qas_id = example.qas_id
                ctx_ex.is_impossible = example.is_impossible
                pyserini_scores[i].append(contexts[j].score)

            new_examples.extend(contexts_examples)
            print(f"\rc: {i}", end='\t')

        # Returns tokenized questions and contexts
        features, dataset = squad_convert_examples_to_features(
            examples=new_examples,
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
        self.new_examples = new_examples
        self.features = features
        self.pyserini_scores = pyserini_scores

    def val_dataloader(self):
        eval_sampler = SequentialSampler(self.dataset)
        eval_dataloader = DataLoader(self.dataset, sampler=eval_sampler,
                                     batch_size=self.hparams.val_batch_size, num_workers=self.hparams.workers)
        return eval_dataloader
