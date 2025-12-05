import os
import torch
from torch import nn
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.decoders import ByteLevel as ByteLevelDecoder, BPEDecoder
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_dataset

from transformer import TransformerEncoderModel



vocab_size = 10_000

batch_size = 64

n_epochs = 20


tokenizer_path = "tokenizers"
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(tokenizer_path, "char_tokenizer.json"),
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)


# Tokenize dataset
def tokenize(batch):
    return hf_tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)



# DataCollator handles masking (15% of tokens)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer,
    mlm=True,
    mlm_probability=0.15
)


# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = TransformerEncoderModel(num_embeddings=vocab_size, d_model=128, padding_idx=hf_tokenizer.pad_token_id, nhead=8, dim_feedforward=4*128, num_layers=4).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")



# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

total_lines = 600_000  # wc -l cc100_combined_subset_shuffled.txt
loss_list = []

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    num_batches = 0

    # Reload dataset for each epoch (streaming dataset is exhausted after one pass)
    dataset = load_dataset('text', data_files={'train': './data/cc100_combined_subset_small_shuffled.txt'}, streaming=True)
    tokenized = dataset['train'].map(tokenize, batched=True, batch_size=batch_size, remove_columns=["text"])
    train_loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator)

    pbar = tqdm(train_loader, total=total_lines//batch_size, desc=f"Epoch {epoch+1}/{n_epochs}")
    for batch in pbar:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(inputs)  # (batch, seq, vocab)

        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        loss_list.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch+1}/{n_epochs} - Avg Loss: {avg_loss:.4f}")

#save model
model.save_path = "/dtu/blackhole/0e/168014/deep-learning/models/transformer_model_char_20.pth"
torch.save(model.state_dict(), model.save_path)

#save loss history as np array
import numpy as np
np.save("training_loss_char_20.npy", np.array(loss_list))