import torch
from torch.utils.data import Dataset, DataLoader


class AnglesDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialize the dataset with features and labels.
        Args:
        - features (np.array): Numpy array of shape (M, N, 12) where M is the number of trajectories,
                               N is the number of steps per trajectory, and 6 is the number of features.
        - labels (np.array): Numpy array of shape (M, N, 4) or (M, 4) .
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, index):
        """Retrieve the nth sample from the dataset."""
        return self.features[index], self.labels[index]
