from bertserini_on_telegram.data import PredictionDataModule
from bertserini_on_telegram.utils.base import Question
from bertserini_on_telegram.utils.pyserini import Searcher
from bertserini_on_telegram.utils.translate import AutoTranslator
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == "__main__":

    # searcher = Searcher("enwiki-paragraphs")
    searcher = Searcher("wikipedia-dpr")
    at = AutoTranslator()

    cli = LightningCLI(run=False, save_config_callback=None)
    bert = cli.model

    question = input("Please input your question[use empty line to exit]:")
    # question = "How big is the Earth?"

    while question != '':
        question, langs = at.translate(question, src_lang=None, trg_lang='en_XX')
        if not question:
            print(f"Sorry, {langs[0]} is not supported by Mbart50 :(")

        print('Translated question: ', question)

        if not question:
            continue

        question = Question(question, "en")
        contexts = searcher.retrieve(question, 10)

        # Create datamodule
        dm = PredictionDataModule(question, contexts, cli.model.hparams.model_name)

        # Predict answer
        cli.trainer.predict(bert, dm)
        answer = bert.answer

        if not answer or len(answer) == 0:
            print(f'Please try changing the phrasing of your question.')
        else:
            answer, _ = at.translate(answer, src_lang='en_XX', trg_lang=langs[0])
            print(f'BERT found an answer to the question! {answer}')
        question = input("Please input your question[use empty line to exit]:")
