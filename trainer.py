from bertserini_pytorch.data import GLUEDataModule
from bertserini_pytorch.models import BERTTrainer

from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == "__main__":
    # run=false is not needed, but used for consistency
    # save_config_overwrite=True is needed to avoid errors
    # when test overwrites fit config file
    cli = LightningCLI(run=False, save_config_overwrite=True)

    # Train model
    cli.trainer.fit(cli.model, cli.datamodule)
