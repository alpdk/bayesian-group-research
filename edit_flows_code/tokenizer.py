"""Byte-level tokenizer for edit-flow code generation.

Token ids 0..255 are raw UTF-8 bytes. Three special ids follow:
BOS and PAD are part of the model vocabulary (the model embeds them),
while GAP only exists in the aligned Z-space used for training targets.
"""

import torch


class ByteTokenizer:
    def __init__(self):
        self.byte_vocab = 256
        self.bos_token = 256
        self.pad_token = 257
        self.vocab_size = 258      # bytes + BOS + PAD (model embedding size)
        self.gap_token = 258       # Z-space only, never fed to the model
        self.num_z_classes = 259   # one-hot classes when interpolating in Z-space

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

    def decode(self, ids, skip_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if skip_special:
            ids = [i for i in ids if i < self.byte_vocab]
        return bytes(ids).decode("utf-8", errors="replace")
