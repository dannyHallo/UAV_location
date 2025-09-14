from torch.utils.data import DataLoader
import os


def get_num_workers():
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count <= 1:
        return 0
    else:
        return cpu_count // 2


class TrajDataLoader(DataLoader):
    """
    A custom DataLoader that pre-configures num_workers and pin_memory.

    This class inherits from torch.utils.data.DataLoader and sets
    default values for num_workers and pin_memory for consistency across
    the project. All other DataLoader arguments can be passed as usual.
    """

    def __init__(self, *args, **kwargs):
        num_workers = get_num_workers()
        print(f"Using {num_workers} worker(s) for data loading.")

        super().__init__(*args, **kwargs, num_workers=num_workers, pin_memory=True)

    def num_batches(self):
        return len(self)

    def num_samples(self):
        return len(self.dataset)
