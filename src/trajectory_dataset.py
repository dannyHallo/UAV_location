import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Stores the trajectory data
    """

    def __init__(self, features, labels, extra_infos):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.extra_infos = torch.tensor(extra_infos, dtype=torch.float32)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, index):
        """Retrieve the nth sample from the dataset."""
        return self.features[index], self.labels[index], self.extra_infos[index]
