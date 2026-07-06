import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions