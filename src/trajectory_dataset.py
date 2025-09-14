import torch
from torch.utils.data import Dataset
import numpy as np


class TrajectoryDataset(Dataset):
    """
    Stores trajectory data and generates valid sequences on-the-fly,
    respecting the boundaries between different trajectories.
    """

    def __init__(
        self, features, labels, extra_infos, trajectory_lengths, sequence_length=1
    ):
        """
        Args:
            features (torch.Tensor): Raw, flat features from all trajectories concatenated.
            labels (torch.Tensor): Raw, flat labels.
            extra_infos (torch.Tensor): Raw, flat extra info.
            trajectory_lengths (list[int]): A list of lengths for each trajectory.
            sequence_length (int): The length of the sequences to generate.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.raw_features = features
        self.raw_labels = labels
        self.raw_extra_infos = extra_infos
        self.trajectory_lengths = trajectory_lengths

        # --- Pre-calculate valid indices ---
        # This is the core logic to prevent cross-trajectory sequences.
        self.valid_indices = []

        # If not using sequences, all indices are valid.
        if self.sequence_length <= 1:
            self.valid_indices = list(range(len(self.raw_features)))
        else:
            current_pos = 0
            for length in self.trajectory_lengths:
                # For a trajectory of a given length, find all possible start points
                # for a sequence of `sequence_length`.
                num_sequences_in_traj = length - self.sequence_length + 1
                if num_sequences_in_traj > 0:
                    # The valid start indices for this trajectory are from `current_pos`
                    # up to `current_pos + num_sequences_in_traj - 1`.
                    self.valid_indices.extend(
                        range(current_pos, current_pos + num_sequences_in_traj)
                    )
                # Move to the start of the next trajectory
                current_pos += length

    def __len__(self):
        """Returns the total number of valid sequences."""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Retrieves the nth valid sequence.

        Args:
            idx (int): The index into the `self.valid_indices` list.

        Returns:
            A tuple of (features, label, extra_info).
        """
        # Get the true starting index in the flat array for the sequence
        start_idx = self.valid_indices[idx]

        if self.sequence_length <= 1:
            # Non-temporal case: just return the single data point
            return (
                self.raw_features[start_idx],
                self.raw_labels[start_idx],
                self.raw_extra_infos[start_idx],
            )
        else:
            # Temporal case: slice the sequence on-the-fly
            end_idx = start_idx + self.sequence_length
            sequence_features = self.raw_features[start_idx:end_idx]

            # The label and extra_info correspond to the last element of the sequence
            target_idx = end_idx - 1
            label = self.raw_labels[target_idx]
            extra_info = self.raw_extra_infos[target_idx]

            return sequence_features, label, extra_info
