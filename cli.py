from bertserini_on_telegram.data import GLUEDataModule
from bertserini_on_telegram.models import BERTTrainer, BERTModule

from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == "__main__":

    cli = LightningCLI()
