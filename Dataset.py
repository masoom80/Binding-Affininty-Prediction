import torch
from torch import tensor
from torch.utils.data import Dataset
import pandas as pd


class BindingAffinityDataset(Dataset):
    def __init__(self, bindings_file):
        self.data = pd.read_csv(bindings_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index, :]
        return tensor(sample.loc[['MHC_sequence_index', 'peptide_sequence_index']]), tensor(sample.loc['label'],
                                                                                            dtype=torch.float32)
