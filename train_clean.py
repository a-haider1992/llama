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
from data_prefetcher import DataPrefetcher
import pdb
import logging

class Model:
    def __init__(self, model, tokenizer, sequence_length=32, logging=None):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = sequence_length
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.logging = logging

    def load_model(self, path):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

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

    def train_model(self, data_prefetcher, num_epochs=50, accumulation_steps=10):
        # Perform model training here
        # pdb.set_trace()
        self.model.train()
        self.model.zero_grad()
        for i in range(num_epochs):
            text = data_prefetcher.next()['text']
            tensor_input = self.tokenizer(text, return_tensors="pt", max_length=self.context_length, truncation=True, padding='max_length')['input_ids']
            # print(f'Input : {text}')
            tensor_input, targets = self.mask_tokens(tensor_input)
            output = self.model(tokens = tensor_input, start_pos = 0)
            loss = self.custom_loss_function(output, targets)
            # clip gradients
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            loss.backward(retain_graph=True)
            print(f'Instance {i + 1}, Loss: {loss.item()}')
        self.optimizer.step()

    def train_model_V2(self, data_prefetcher, num_epochs=50, accumulation_steps=10, max_gradient_norm=None, mask_prob=0.15):
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0

            for iteration in range(accumulation_steps):
                batch = data_prefetcher.next()
                text = batch['text']
                tensor_input = self.tokenizer(text, return_tensors="pt", max_length=self.context_length, truncation=True, padding='max_length')['input_ids']
                tensor_input, targets = self.mask_tokens(tensor_input, mask_prob=mask_prob)

                output = self.model(tokens=tensor_input, start_pos=0)
                loss = self.custom_loss_function(output, targets)
                total_loss += loss.item()

                loss.backward(retain_graph=True)

                # Optionally clip gradients
                if max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_gradient_norm)

            total_loss /= accumulation_steps  # Average loss over accumulation steps
            print(f'Epoch {epoch + 1}, Average Loss: {total_loss}')
            self.logging.info(f'Epoch {epoch + 1}, Average Loss: {total_loss}')

        # Perform the final optimization step after the entire training loop
        self.optimizer.step()
        self.model.zero_grad()

        
def main():
    # Load datasets from the datasets library
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'}) # add new mask token to tokenizer
    data = load_dataset('bookcorpus', streaming=True)
    sequence_length = 64

    data_prefetcher = DataPrefetcher(loader=data['train'])

    # Initialize model parallel
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    initialize_model_parallel(2)
    torch.cuda.set_device(dist.get_rank())
    torch.autograd.set_detect_anomaly(True)
    model_args = ModelArgs(dim=1024, vocab_size=tokenizer.vocab_size, n_layers=1, n_heads=1, max_seq_len=sequence_length,)
    pdb.set_trace()

    # set up logging
    log_file_name = 'training.log'
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_name)

    # model = DDP(model, device_ids=[0], output_device=0)
    for i in range(10):
        logging.info(f'Model training iteration: {i + 1}')
        print(f'Model training iteration: {i + 1}')
        m = Model(model=Transformer_modified(model_args), tokenizer=tokenizer, logging=logging)
        logging.info(f'Model has {m.count_parameters():.1f}M trainable parameters.')
        # print(f'Model has {m.count_parameters():.1f}M trainable parameters.')
        m.load_model(f'llama-scratch.pt')
        m.train_model_V2(data_prefetcher, num_epochs=1000)
        m.save_model(f'llama-scratch.pt')
        print('Model saved!')
        logging.info('Model saved!')
        del m
        print('----------------------')
    # m.train_model_V2(data_prefetcher, num_epochs=10, mask_prob=0.20)
    # torch.save(m.model.state_dict(), 'llama-scratch.pt')
    # clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
