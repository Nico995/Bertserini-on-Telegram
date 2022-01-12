from torch.utils.data import Dataset
import pandas as pd


class ShARCDataset(Dataset):
    def __init__(self, data_path:str):
        super().__init__()

        self.data_path = data_path
        self.data = pd.read_json(data_path)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        return dict(self.data.iloc[index])
