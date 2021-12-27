from bertserini_pytorch.data import GLUEDataModule
from bertserini_pytorch.models import BERTTrainer, BERTModule

from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == "__main__":

    cli = LightningCLI()
