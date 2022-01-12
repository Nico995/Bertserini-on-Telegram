from bertserini_on_telegram.data.datamodules import ShARCDataModule, SQuADDataModule

# dm = ShARCDataModule(data_root='/home/nicola/data/sharc/json')
dm = SQuADDataModule('deepset/bert-base-cased-squad2')

dm.prepare_data()
dm.setup("validate")
print(next(iter(dm.train_dataloader())))