import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from aminoacid import to_onehot
from file_utils import read_go_id

class CustomDataset(Dataset):
    def __init__(self, go_id, data_path):
        super().__init__()
        self.idx_map = go_id
        self.data = pd.read_pickle(data_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence_data = self.data.iloc[idx]
        seq = to_onehot(sequence_data['sequence'])
        seq_embed = sequence_data['embedding']
        annotations = sequence_data['annotations']

        cls = np.array([1 if a.strip() in self.idx_map else 0 for a in annotations], dtype=np.float32)
        return np.transpose(seq), seq_embed, np.transpose(cls)

    def __len__(self):
        return len(self.data)

def load_datasets(args, go_id):
    datasets = {}
    for phase in ['train', 'val', 'test']:
        dir_path = Path(getattr(args, f'{phase}_data_dir'))
        dataset = CustomDataset(go_id, dir_path)
        datasets[phase] = DataLoader(dataset=dataset, batch_size=args.batch_size)

    go_list = read_go_id(Path(args.go_SS_dir))
    return datasets['train'], datasets['val'], datasets['test'], go_list
