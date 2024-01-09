from datasets import load_dataset
import llama
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, pipeline
import pdb

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel
)

MODEL = "meta-llama/Llama-2-7b-chat-hf"
MODEL_tiny = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('gpt2')

# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # add new pad token to tokenizer

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count = count / 1000**2
    return count

# data = load_dataset("oscar", "unshuffled_deduplicated_en", streaming=True)
data = load_dataset("bookcorpus",  streaming=True)
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

def mask_tokens(input_ids, tokenizer, mask_prob=0.15):
    """
    Randomly masks tokens in the input_ids tensor for masked language modeling pretraining.

    Args:
        input_ids (torch.Tensor): Tensor containing input token IDs.
        tokenizer: Pretrained tokenizer.
        mask_prob (float): Probability of masking a token.

    Returns:
        (torch.Tensor, torch.Tensor): Masked input tensor, labels tensor.
    """
    mask = torch.rand(input_ids.shape) < mask_prob
    masked_ids = input_ids.clone()
    masked_ids[mask] = tokenizer.mask_token_id

    # Prepare labels tensor for computing loss
    labels = torch.full_like(input_ids, fill_value=-100)  # -100 is the default value for ignored index in cross-entropy loss
    labels[mask] = input_ids[mask]

    return masked_ids, labels

class ConstantLengthDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
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
    # def __iter__(self):
    #     return self.tokenizer(next(iter(self.data['train']))['text'], return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')
    def __getitem__(self, idx):
        data = self.tokenizer(next(iter(self.data['train']))['text'], return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')
        return data
    
def compute_loss(logits, labels):
    """
    Computes cross-entropy loss between logits and labels.

    Args:
        logits (torch.Tensor): Tensor containing logits.
        labels (torch.Tensor): Tensor containing labels.

    Returns:
        torch.Tensor: Scalar tensor containing the loss.
    """
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

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
seq_len = 12

# instantiate the constant length dataset
data_const = ConstantLengthDataset(data=data, tokenizer=tokenizer, max_length=seq_len)
# print(data_const[0]['input_ids'].size())

data_loader = DataLoader(data_const, batch_size=1, shuffle=True)
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
model_args = llama.ModelArgs(vocab_size=tokenizer.vocab_size, n_layers=4, n_heads=2, max_seq_len=12,)

# Parallelize the model
torch.distributed.init_process_group(backend="nccl")
initialize_model_parallel(1)
torch.cuda.set_device(0)

# build an empty model
model = llama.Transformer(model_args)
model_tiny = AutoModelForCausalLM.from_pretrained(MODEL_tiny)
print(f"Number of parameters in the model: {count_parameters(model_tiny):.1f}M")
pdb.set_trace()
# print(model_tiny)

# count the number of parameters in the model
print(f"Number of parameters in the model: {count_parameters(model):.1f}M")
# print(model)
# pdb.set_trace()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch = 0
while epoch < 5:
    for item in data_loader:
        print(item)
        input_ids = item['input_ids']

        # Use mask_tokens function to create masked input and labels
        masked_input_ids, labels = mask_tokens(input_ids, tokenizer)

        # Forward pass
        logits = model(tokens = masked_input_ids.squeeze(0), start_pos = 0)
        # print(logits)

        # Compute loss
        loss = compute_loss(logits, labels)
        print(f"Epoch {epoch} loss: {loss.item()}")

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
#         eval_dataset = Book_data,
#         train_dataset = Book_data)

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





