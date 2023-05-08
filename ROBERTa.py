import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np

# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

# Load the IMDB dataset
df = pd.read_csv('IMDB_Dataset.csv')

# Map the labels to 0 and 1 (negative and positive)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# Tokenize the text data and convert it to tensors
encoded_data_train = tokenizer.batch_encode_plus(df[df.type=='train'].review.values,
                                                  add_special_tokens=True,
                                                  return_attention_mask=True,
                                                  pad_to_max_length=True,
                                                  max_length=256,
                                                  return_tensors='pt')

encoded_data_val = tokenizer.batch_encode_plus(df[df.type=='test'].review.values,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                pad_to_max_length=True,
                                                max_length=256,
                                                return_tensors='pt')

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.type=='train'].sentiment.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.type=='test'].sentiment.values)

# Create a dataset from the encoded data
dataset_train = torch.utils.data.TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = torch.utils.data.TensorDataset(input_ids_val, attention_masks_val, labels_val)

# Create a dataloader for the training and validation sets
batch_size = 32
dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)

# Load the pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, output_attentions=False, output_hidden_states=False)

# Set the device to run on (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the optimizer and the learning rate scheduler
epochs = 2
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(dataloader_train) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# Train the RoBERTa model
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for batch in dataloader_train:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_masks, labels = batch
        model.zero_grad()
        outputs = model(input_ids, 
                        attention_mask=attention_masks, 
                        labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
       
