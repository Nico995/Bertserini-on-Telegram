from bertserini_pytorch.data.datamodules import SQuADDataModule
from bertserini_pytorch.models.modules import BERTPredictor
from pytorch_lightning import Trainer
dm = SQuADDataModule('squad_v2')
dm.prepare_data()
dm.setup("validate")
model = BERTPredictor()

trainer = Trainer(gpus=1)
trainer.validate(model, dm)
