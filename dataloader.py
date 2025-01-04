import numpy as np
import os
import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, verbose=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}
        self.rng = np.random.default_rng(1337)

        self.enc = tiktoken.get_encoding("gpt2")

        # get the shard filenames
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f'no shards found for split {split}'
        if verbose:
            print(f'found {len(shards)} shards for split {split}')
        self.reset()

    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)  # Github Errata
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def load_shard(self, filename): # added from PR to avoid periodisation in training: https://github.com/karpathy/build-nanogpt/pull/52/files
        shard = self.load_tokens(filename)
        if self.split == 'train':
            # split tokens into documents using the <|endoftext|> special token and shuffle
            eot_positions = (torch.where(shard == self.enc.eot_token)[0] + 1).tolist()
            documents = [shard[start:end] for start, end in zip([0] + eot_positions[:-1], eot_positions)]
            self.rng.shuffle(documents)
            shard = torch.cat(documents) # concatenate the documents back together
        return shard

    def set(self, loader_checkpoint):
        B, T = self.B, self.T
        self.current_position = loader_checkpoint['current_position'] + self.B * self.T * self.process_rank # we add the B*T*process_rank to the position to make sure it is the correct position for each process
        self.current_shard = loader_checkpoint['current_shard']
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard += 1
            # reshuffle after each epoch
            if self.current_shard == len(self.shards):
                self.reset()
            else:
                self.tokens = self.load_shard(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        if self.split == 'train':
            self.rng.shuffle(self.shards)
        self.tokens = self.load_shard(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + (B * T + 1)]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard += 1
            # reshuffle after each epoch
            if self.current_shard == len(self.shards):
                self.reset()
            else:
                self.tokens = self.load_shard(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
