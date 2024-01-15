from datasets import load_dataset
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel
)
from llama.model_modified import Transformer_modified, ModelArgs
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import pdb

class Model:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def custom_loss_function(self, outputs, labels):
        labels = labels.view(-1)
        outputs = F.softmax(outputs, dim=-1)
        outputs = outputs.view(-1, outputs.size(-1))
        loss = F.cross_entropy(outputs, labels, reduction='mean')
        return loss
    
    def mask_tokens(self, input_tensor, mask_prob=0.15):
        """
        Randomly masks tokens in the input tensor for masked language modeling pretraining.

        Args:
            input_tensor (torch.Tensor): Tensor containing input token IDs.
            tokenizer: Pretrained tokenizer.
            mask_prob (float): Probability of masking a token.

        Returns:
            (torch.Tensor, torch.Tensor): Masked input tensor, labels tensor.
        """
        mask = torch.rand(input_tensor.shape) < mask_prob

        # Perform masking operation in-place
        input_tensor[mask] = self.tokenizer.mask_token_id

        # Prepare labels tensor for computing loss
        labels = torch.full_like(input_tensor, fill_value=-100)
        labels[mask] = input_tensor[mask]

        return input_tensor, labels
    
    def count_parameters(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        count = count / 1000**2
        return count

    def train_model(self, data_loader):
        # Perform model training here
        for (tensor_input, text) in data_loader:
            print(f'Input : {text}')
            self.optimizer.zero_grad()
            # tensor_input, targets = self.mask_tokens(self.tokenizer("Hello how are you, I m doing great..", return_tensors="pt", max_length=32, truncation=True, padding='max_length')['input_ids'])
            tensor_input, targets = self.mask_tokens(tensor_input.squeeze(0))
            # # Forward pass
            # print(f'Input : {tensor_input} has grad : {tensor_input.requires_grad}')
            output = self.model(tokens = tensor_input, start_pos = 0)
            # print(f'Output : {output} has grad : {output.requires_grad}')
            # # Loss computation
            loss = self.custom_loss_function(output, targets)
            print(f'Loss : {loss.item()}')
            # print(f'Loss has grad : {loss.requires_grad}')
            # print(output._version)
            # ## Backpropagation
            loss.backward(retain_graph=True)
            # Weight updates - with retain_graph=True, we can call backward multiple times, however, we need to call optimizer.step() only once
        self.optimizer.step()

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.sequence_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get data at the specified index
        self.input_ids = self.tokenizer(self.data['train'][index]['text'], return_tensors="pt", max_length=self.sequence_length, truncation=True, padding='max_length')['input_ids']
        return self.input_ids, self.data['train'][index]['text']
    
class CustomIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.sequence_length = max_length
        self.index = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        instance = self.data['train'][self.index]
        self.index += 1
        input_ids = self.tokenizer(instance['text'], return_tensors="pt", max_length=self.sequence_length, truncation=True, padding='max_length')['input_ids']
        yield input_ids, instance['text']
        
def main():
    # Load datasets from the datasets library
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data = load_dataset('bookcorpus')
    sequence_length = 32
    dataset = CustomDataset(data=data, tokenizer=tokenizer, max_length=sequence_length)
    dataset_iterable = CustomIterableDataset(data=data, tokenizer=tokenizer, max_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    data_loader_iterable = DataLoader(dataset_iterable, batch_size=1)

    # Initialize model parallel
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    initialize_model_parallel(1)
    torch.autograd.set_detect_anomaly(True)
    model_args = ModelArgs(dim=32, vocab_size=tokenizer.vocab_size, n_layers=1, n_heads=1, max_seq_len=sequence_length,)
    # pdb.set_trace()
    m = Model(model=Transformer_modified(model_args), tokenizer=tokenizer)
    print(f'Model has {m.count_parameters():.1f}M trainable parameters.')
    # model = DDP(model, device_ids=[0], output_device=0)
    m.train_model(data_loader)
    # clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
