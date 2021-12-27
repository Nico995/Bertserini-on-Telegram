from bertserini_on_telegram.data import PredictionDataModule
from bertserini_on_telegram.utils.base import Question
from bertserini_on_telegram.utils.pyserini import Searcher
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == "__main__":

    # searcher = build_searcher("enwiki-paragraphs")
    searcher = Searcher("wikipedia-dpr")

    cli = LightningCLI(run=False, save_config_callback=None)
    bert = cli.model
    question = "When did Beyonce start becoming popular?"
    # context = "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'."
    while True:
        # question = input("Please input your question[use empty line to exit]:")
        question = Question(question, "en")
        contexts = searcher.retrieve(question, 20)
        dm = PredictionDataModule(question, contexts, cli.model.hparams.model_name)
        # contexts = [Context(text=context, score=1)]
        cli.trainer.predict(bert, dm)
        # answer = bert.predict_combination(question, contexts)
        print(f'BERT found an answer to the question! {bert.answer}')
        question = input("Please input your question[use empty line to exit]:")
