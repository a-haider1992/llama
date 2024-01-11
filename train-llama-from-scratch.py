from datasets import load_dataset
import llama
from llama.model_modified import Transformer_modified
from llama.model import Transformer
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, pipeline
import pdb

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel
)

import torch.nn.functional as F
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

MODEL = "meta-llama/Llama-2-7b-chat-hf"
MODEL_tiny = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('gpt2')

# tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # add new pad token to tokenizer
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'}) # add new mask token to tokenizer

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count = count / 1000**2
    return count

# data = load_dataset("oscar", "unshuffled_deduplicated_en", streaming=True)
data = load_dataset("bookcorpus", streaming=True)
# pdb.set_trace()

# torch_data = data.with_format("torch")

# class OscarDataset(Dataset):
#     def __init__(self, data, tokenizer):
#         self.data = data
#         self.tokenizer = tokenizer
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         data = self.tokenizer(self.data['train'][idx]['text'], return_tensors="pt")
#         return data

def mask_tokens(input_tensor, tokenizer, mask_prob=0.15):
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
    masked_tensor[mask] = tokenizer.mask_token_id

    # Prepare labels tensor for computing loss
    labels = torch.full_like(input_tensor, fill_value=-100)  # -100 is the default value for ignored index in cross-entropy loss
    labels[mask] = input_tensor[mask]

    return masked_tensor, labels

class ConstantLengthDataset(IterableDataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.data_iter = iter(data['train'])
        self.input_ids = []
        self.attention_mask = []
        self.max_length = max_length
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    # def __getitem__(self, idx):
    #     self.input_ids = self.tokenizer(self.data['train'][idx]['text'], return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')['input_ids']
    #     self.attention_mask = self.tokenizer(self.data['train'][idx]['text'], return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')['attention_mask']
    #     return {'input_ids': self.input_ids, 'attention_mask': self.attention_mask}
    def __iter__(self):
        for data_instance in self.data['train']:
            text = data_instance['text']
            data = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')
            input_ids = data['input_ids']
            yield mask_tokens(input_ids, self.tokenizer)
            # yield {'input_ids': input_ids, 'attention_mask': data['attention_mask']}
        # return self.tokenizer(next(iter(self.data['train']))['text'], return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')
    # def __getitem__(self, idx):
    #     text = self.data['train'][idx]['text']
    #     data = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')
    #     input_ids = data['input_ids'].squeeze(0)
    #     return {'input_ids': input_ids, 'attention_mask': data['attention_mask'].squeeze(0)}
        # pdb.set_trace()
        # try:
        #     # Get the next data instance from the iterator
        #     data_instance = next(self.data_iter)
        # except StopIteration:
        #     # If the iterator is exhausted, reset it
        #     self.data_iter = iter(self.data['train'])
        #     data_instance = next(self.data_iter)
        # # print(data_instance['text'])
        # data = self.tokenizer(data_instance['text'], return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')
        # # print(data)
        # # return {'tokens': data['input_ids'].squeeze(0), 'start_pos': 0, 'labels': data['input_ids'].squeeze(0)}
        # return data
    
def compute_loss(logits, labels, mask_token_id=-100):
    """
    Computes cross-entropy loss between logits and labels, ignoring padding tokens.

    Args:
        logits (torch.Tensor): Tensor containing logits.
        labels (torch.Tensor): Tensor containing labels.
        padding_token_id (int): Token ID for padding (default is -100).

    Returns:
        torch.Tensor: Scalar tensor containing the loss.
    """
    # Flatten logits and labels
    # pdb.set_trace()
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)

    # pdb.set_trace()

    # Create a mask to ignore padding tokens
    mask = (labels_flat != mask_token_id)

    # Compute the cross-entropy loss, ignoring padding tokens
    # loss = F.cross_entropy(logits_flat, labels_flat)
    loss = F.cross_entropy(logits_flat[mask], labels_flat[mask])

    return loss

# write custom torch dataset class using loaded data object from datasets library
# class BookCorpusDataset(torch.utils.data.IterableDataset):
#     def __init__(self, data, tokenizer):
#         self.data = data
#         self.tokenizer = tokenizer
#     def __iter__(self):
#         data = yield self.tokenizer(next(iter(self.data['train']))['text'], return_tensors="pt")['input_ids']
#         return data
#     # Implement __len__ so that TorchDataPipe can automatically infer the length
#     def __len__(self):
#         return len(self.data)
    
# pdb.set_trace()
# Book_data = BookCorpusDataset(data=data, tokenizer=tokenizer)
# Book_data = OscarDataset(data=data, tokenizer=tokenizer)
# book_loader = DataLoader(Book_data, batch_size=1, shuffle=True)
# print(f'Book data element is: {Book_data[0]}')
# print(f"Length of book corpus each element is: {Book_data[0]['input_ids'].squeeze().size()}")
# print(tokenizer(data['train'][0]['text']).tokens())
# print(f"Number of tokens in first data instance: {len(tokenizer(data['train'][0]['text']).tokens())}")
# print(f'Vocab size: {tokenizer.vocab_size}')

# pdb.set_trace()

# count the number of tokens in the raw dataset
# data_iter = iter(data['train'])
# total_tokens = 0
# for element in data_iter:
#     # print(element['text'])
#     total_tokens += len(tokenizer(element['text']).tokens())

# print(f"Total number of tokens present in the raw dataset : {total_tokens}")

total_sequences = 0
seq_len = 100

# instantiate the constant length dataset
data_const = ConstantLengthDataset(data=data, tokenizer=tokenizer, max_length=seq_len)
# print(data_const[0]['input_ids'].size())

batch_size = 1
dataloader = DataLoader(data_const, batch_size=batch_size, num_workers=1, collate_fn=lambda x: x[0])

# Initialize an empty list to store the tensors
tensor_list = []
pdb.set_trace()

max_batches = 32
batch_count = 0

# Iterate over the dataloader and store the tensors
# for batch in dataloader:
#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     tensor_list.append(input_ids)

#     # Increment the batch count
#     batch_count += 1

#     # Break the loop if the desired number of batches is reached
#     if batch_count >= max_batches:
#         break

# data_const_iter = iter(data_const)

#count the number of sequences in the constant length dataset
# for element in data_const:
#     # inputs = element['input_ids'].tolist()
#     # print(tokenizer.decode(inputs[0]))
#     total_sequences += element['input_ids'].size()[1]

# print(f"Total number of seqeunces of size {seq_len} by constant length Dataset is : {total_sequences}")
# print(f" Total number of tokens in the constant length dataset is : {(total_sequences * seq_len)/ (1000**2):.1f}M")


# acces Book_data via __iter__ method
# print(type(next(iter(Book_data))))
# print(next(iter(Book_data)))
# print(f'book corpus data is an instance of torch dataset {isinstance(Book_data, torch.utils.data.IterableDataset)}')

# torch_data = IterableWrapper(torch_data)
# print(f'book corpus data is an instance of torch dataset {isinstance(torch_data, torch.utils.data.IterableDataset)}')

# eval_data = load_dataset('rotten_tomatoes', "unshuffled_deduplicated_en")

# Using the pre-trained tokenizer from the llama package


# tokenized_data = torch_data.map(lambda x: tokenizer(x["text"], return_tensors="pt"))
# print(f'Tokenize book corpus data is an instance of torch dataset {isinstance(tokenized_data, torch.utils.data.IterableDataset)}')

# tokenized_eval_data = eval_data.map(lambda x: tokenizer(x["text"], return_tensors="pt"))
# torch_eval_data = tokenized_eval_data.with_format("torch")
# assert isinstance(torch_eval_data, torch.utils.data.Dataset)

# set model arguments
# dim controls the dimensionality of the model
model_args = llama.ModelArgs(dim=512, vocab_size=tokenizer.vocab_size, n_layers=4, n_heads=4, max_seq_len=seq_len,)

# Parallelize the model
torch.distributed.init_process_group(backend="nccl")
initialize_model_parallel(1)
torch.autograd.set_detect_anomaly(True)
torch.cuda.set_device(0)

# build an empty model
model = Transformer_modified(model_args)
# model = Transformer(model_args)
# model_tiny = AutoModelForCausalLM.from_pretrained(MODEL_tiny)
# print(f"Number of parameters in the model: {count_parameters(model_tiny):.1f}M")
# pdb.set_trace()
# print(model_tiny)

# count the number of parameters in the model
print(f"Number of parameters in the model: {count_parameters(model):.1f}M")
# print(model)
# pdb.set_trace()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# epoch = 0
# train the model
# model = model.to(torch.device('cuda'))
# pdb.set_trace()
# Stack the tensors to create the final tensor of shape [samples, seq_len, vocab_size]
# final_tensor = torch.stack(tensor_list, dim=0)

# print(f"Final tensor shape: {final_tensor.shape}")

# training code below

# process group init
# dist_init(rank, world_size)
pdb.set_trace()
rank = 0

# Problem statement
model = model.to(rank)

# optimizer specific arguments e.g. LR, momentum, etc...
base_optimizer_arguments = { "lr": 1e-4}

# Wrap a base optimizer into OSS
base_optimizer = torch.optim.SGD  # any pytorch compliant optimizer
optimizer = OSS(
    params=model.parameters(),
    optim=base_optimizer,
    **base_optimizer_arguments)

# Wrap the model into ShardedDDP, which will reduce gradients to the proper ranks
model = ShardedDDP(model, optimizer)

# Any relevant training loop, nothing specific to OSS. For example:
model.train()
for e in range(1):
    total_loss = 0.0
    count = 0.0
    for (data, target) in dataloader:
        # print(data)
        # print(target)
        count = count + 1
        data, target = data.to(rank), target.to(rank)
        # Train
        # model.zero_grad()
        outputs = model(data, start_pos = 0)
        loss = compute_loss(outputs, target)
        total_loss += loss
        print(f"loss: {loss.item()}")
        if count >= 100:
            break
    print(f"Epoch {e + 1} total loss: {total_loss.item()}")
    pdb.set_trace()
    loss_val = total_loss / count
    model.zero_grad()
    loss_val.backward()
    optimizer.step()    

# for epoch in range(1):
#     # Use mask_tokens function to create masked input and labels
#     for batch in dataloader:
#         print(batch['input_ids'].shape)
#         masked_input_ids, labels = mask_tokens(batch['input_ids'], tokenizer)

#         pdb.set_trace()

#         # Forward pass
#         logits = model(tokens = masked_input_ids.clone(), start_pos = 0)
#         # print(logits)

#         # Compute loss
#         loss = compute_loss(logits, labels, mask_token_id=tokenizer.mask_token_id)
#         optimizer.zero_grad()  # Clear existing gradients
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         print(f"Epoch {epoch + 1} loss: {loss.item()}")
        # torch.save(model.state_dict(), f"model_scratch_{epoch}.pt")

# text = "I like to eat"
# output = model(tokens = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True, padding='max_length')['input_ids'], start_pos = 0)
# print(tokenizer.decode(output[0].tolist()))

    # batch = next(iter(data_const))
    # original_tokens = batch['input_ids'].squeeze()[-1].item()  # Store the value as it is
    # squeezed_batch = batch['input_ids'].squeeze().tolist()  # Convert to list to modify
    # squeezed_batch[-1] = tokenizer.mask_token_id
    # inputs = torch.tensor(squeezed_batch).unsqueeze(0)  # Convert back to tensor and unsqueeze
    # # print(batch['input_ids'])
    # # print(batch['attention_mask'])
    # # print(batch['labels'])
    # # print(batch['input_ids'].shape)
    # # print(batch['attention_mask'].shape)
    # logits = model_tiny(tokens=inputs, start_pos = 0)
    # print(tokenizer.decode(inputs[0].tolist()))
    # output = torch.round(logits[0].T[0]).to(torch.int)
    # print(tokenizer.decode(output.tolist()))
    # next_token = torch.argmax(logits[:, -1], dim=-1)
    # actual_token = torch.tensor(original_tokens).unsqueeze(0).to(torch.long)
    # loss = torch.nn.functional.cross_entropy(logits[:, -1].clone(), actual_token.clone())
    # optimizer.zero_grad()  # Clear existing gradients
    # print(f"Epoch {epoch} loss: {loss.item()}")
    # loss.backward(retain_graph=True)
    # optimizer.step()

# write training loop and compute loss using the model and Book_data


# def train(model, data):
#     for batch in data:
#         # print(batch)
#         # print(batch['input_ids'])
#         # print(batch['attention_mask'])
#         # print(batch['labels'])
#         # print(batch['input_ids'].shape)
#         # print(batch['attention_mask'].shape)
#         outputs = model(tokens=batch['input_ids'], start_pos = 0)
#         loss = outputs.loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# def __main__():
#     pdb.set_trace()
#     train(model, Book_data)

# __main__()

# training_args = TrainingArguments(
#                                   save_steps = 5000,
#                                   warmup_steps = 10,
#                                   logging_steps = 100,
#                                   weight_decay = 0.05,
#                                   num_train_epochs = 1,
#                                   logging_dir = './logs',
#                                   output_dir = './results',
#                                   per_device_eval_batch_size = 1,
#                                   per_device_train_batch_size = 1)

# pdb.set_trace()
# # Training script 
# trainer = Trainer(model = model,
#         args = training_args,
#         eval_dataset = data_const,
#         train_dataset = data_const)

# trainer.train()




# print(next(iter(data)))
# train_data, test_data, _, _ = data.train_test_split(test_size=0.1)

# Model = AutoModelForCausalLM.from_pretrained("llama")
# Tokenizer = AutoTokenizer.from_pretrained("llama")


# print(tokenizer(next(iter(data))['text']))
# pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# pipeline = pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# sequences = pipeline("Explain deep learning.", max_length = 30, num_return_sequences=3)
# print(f"Result: {sequences}")





