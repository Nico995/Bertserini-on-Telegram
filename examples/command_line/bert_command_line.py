from bertserini_on_telegram.data import PredictionDataModule
from bertserini_on_telegram.utils.base import Question
from bertserini_on_telegram.utils.pyserini import Searcher
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == "__main__":

    # searcher = build_searcher("enwiki-paragraphs")
    searcher = Searcher("wikipedia-dpr")

    cli = LightningCLI(run=False, save_config_callback=None)
    bert = cli.model
        
    question = input("Please input your question[use empty line to exit]:")

    while question != '':
        question = Question(question, "en")
        contexts = searcher.retrieve(question, 20)
        dm = PredictionDataModule(question, contexts, cli.model.hparams.model_name)
        
        cli.trainer.predict(bert, dm)
        print(f'BERT found an answer to the question! {bert.answer}')
        question = input("Please input your question[use empty line to exit]:")
