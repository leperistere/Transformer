import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input text
input_text = "Hello, how are you?"
input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)  # Batch size 1

# Forward pass through BERT model
outputs = model(input_ids)

# Get the output embeddings for each token in the input text
last_hidden_states = outputs[0]

# Print the output embeddings for each token
for token_embedding in last_hidden_states[0]:
    print(token_embedding)
