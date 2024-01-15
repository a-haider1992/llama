from datasets import load_dataset
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel
)
from llama.model_modified import Transformer_modified, ModelArgs
from torch.utils.data import Dataset, DataLoader
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
        softmax_outputs = F.softmax(outputs, dim=-1)

        # Flatten both the softmax_outputs and labels
        softmax_outputs_flat = softmax_outputs.view(-1, softmax_outputs.size(-1))
        labels_flat = labels.view(-1)

        # Compute the cross-entropy loss
        loss = F.cross_entropy(softmax_outputs_flat, labels_flat, reduction='mean')
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
        masked_tensor = input_tensor.clone()
        masked_tensor[mask] = self.tokenizer.mask_token_id

        # Prepare labels tensor for computing loss
        labels = torch.full_like(input_tensor, fill_value=-100)  # -100 is the default value for ignored index in cross-entropy loss
        labels[mask] = input_tensor[mask]

        return masked_tensor, labels
    
    def count_parameters(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        count = count / 1000**2
        return count

    def train_model(self, data_loader):
        # Perform model training here
        for index, (tensor_input, text) in enumerate(data_loader):
            print(f'Index : {index+1}')
            # print(f'Input : {text}')
            self.optimizer.zero_grad()
            # tensor_input, targets = self.mask_tokens(self.tokenizer("Hello how are you, I m doing great..", return_tensors="pt", max_length=32, truncation=True, padding='max_length')['input_ids'])
            tensor_input, targets = self.mask_tokens(tensor_input.squeeze(0))
            # # Forward pass
            print(f'Input : {tensor_input} has grad : {tensor_input.requires_grad}')
            output = self.model(tokens = tensor_input, start_pos = 0)
            print(f'Output : {output} has grad : {output.requires_grad}')
            # # Loss computation
            loss = self.custom_loss_function(output, targets)
            print(f'Loss : {loss.item()}')
            print(f'Loss has grad : {loss.requires_grad}')
            # print(output._version)
            # ## Backpropagation
            loss.backward()
            # # Weight updates
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
        
def main():
    # Load datasets from the datasets library
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data = load_dataset('rotten_tomatoes')
    sequence_length = 32
    dataset = CustomDataset(data=data, tokenizer=tokenizer, max_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

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
