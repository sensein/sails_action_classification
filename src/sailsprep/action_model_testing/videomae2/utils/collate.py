import torch


def collate(batch):
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels, dtype=torch.long)
