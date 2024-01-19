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
from typing import List, Literal, Optional, Tuple, TypedDict


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class Model:
    def __init__(self, model, tokenizer, sequence_length=32, logging=None, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = sequence_length
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.logging = logging
        self.device = device

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
                self.logging.info(f'Input : {text}')
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

# Model inference class
class ModelInference:
    def __init__(self, model, tokenizer, sequence_length=32, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = sequence_length
        self.device = device

    def load_model(self, path):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
    
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

        
def main():
    # Load datasets from the datasets library
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'}) # add new mask token to tokenizer
    data = load_dataset('bookcorpus', streaming=True)
    sequence_length = 32

    # Initialize model parallel
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    initialize_model_parallel(2)
    torch.cuda.set_device(dist.get_rank())
    torch.autograd.set_detect_anomaly(True)
    model_args = ModelArgs(dim=32, vocab_size=tokenizer.vocab_size, n_layers=1, n_heads=1, max_seq_len=sequence_length,)
    pdb.set_trace()

    # set up logging
    log_file_name = 'training.log'
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_name)

    # model = DDP(model, device_ids=[0], output_device=0)
    for i in range(100):
        logging.info('----------------------')
        logging.info(f'Model training iteration: {i + 1}')
        print(f'Model training iteration: {i + 1}')
        data_prefetcher = DataPrefetcher(loader=data['train'])
        m = Model(model=Transformer_modified(model_args), tokenizer=tokenizer, logging=logging, device=device)
        logging.info(f'Model has {m.count_parameters():.1f}M trainable parameters.')
        # print(f'Model has {m.count_parameters():.1f}M trainable parameters.')
        m.load_model(f'llama-scratch.pt')
        m.train_model_V2(data_prefetcher, num_epochs=10, max_gradient_norm=1.0, mask_prob=0.15)
        m.save_model(f'llama-scratch.pt')
        print('Model saved!')
        logging.info('Model saved!')
        del m
        del data_prefetcher
        logging.info('----------------------')
    # Model Inferece
    # model = ModelInference(model=Transformer_modified(model_args), tokenizer=tokenizer, device=device)
    # model.load_model(f'llama-scratch.pt')
    # print(model.text_completion(['The cat sat on the mat.']))
    # clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
