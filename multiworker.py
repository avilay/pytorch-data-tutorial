import os
from datetime import datetime
import numpy as np
from cprint import cprint
import torch.utils.data as td

rng = np.random.default_rng()


def log(msg):
    pid = os.getpid()
    now = datetime.now().strftime("%H:%M:%S")
    wi = td.get_worker_info()
    color = wi.id + 1 if wi is not None else 0
    worker_info = f"{wi.id+1}/{wi.num_workers}" if wi is not None else ""
    cprint(color, f"{now} [{pid}] {worker_info} - {msg}")


class MyMappedDataset(td.Dataset):
    def __init__(self, n=3, m=20):
        self._m = m
        self._n = n

    def __getitem__(self, idx):
        log(f"MyMappedDataset: Fetching data[{idx}]")
        X = np.full((self._n,), fill_value=idx, dtype=np.float32)
        y = rng.choice([0.0, 1.0]).astype(np.float32)
        return X, y

    def __len__(self):
        return self._m


class MyBatcher(td.Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self._batch_sampler = td.BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        for batch in self._batch_sampler:
            log(f"MyBatcher: Yielding batch {batch}")
            yield batch


def main():
    ds = MyMappedDataset()
    dl = td.DataLoader(
        ds,
        batch_sampler=MyBatcher(
            td.SequentialSampler(ds), batch_size=3, drop_last=False
        ),
        num_workers=2,
        prefetch_factor=2,
    )
    for i, (X, y) in enumerate(dl):
        log(f"\nX={X}\ny={y}")
        input("Press ENTER to get the next batch...")


if __name__ == "__main__":
    main()
