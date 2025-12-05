import os
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding
from tqdm import tqdm

from transformer import TransformerEncoderModel, TransformerClassifierModel


model_path = "./models/transformer_model_char_20.pth"
vocab_size = 10_000
n_classes = 4

batch_size = 64
n_epochs = 100

# Early stopping parameters
patience = 3  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change to qualify as an improvement

tokenizer_path = "tokenizers"
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(tokenizer_path, "char_tokenizer.json"),
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)


dataset = load_dataset("ag_news")

#split train into train and val set
dataset_train = dataset['train'].train_test_split(test_size=0.1)

# Tokenize dataset
def tokenize(batch):
    return hf_tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)

tokenized = dataset_train['train'].map(tokenize, batched=True, batch_size=batch_size, remove_columns=["text"])

tokenized_val = dataset_train['test'].map(tokenize, batched=True, batch_size=batch_size, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=hf_tokenizer, return_tensors="pt")

# Create DataLoader
train_loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(tokenized_val, batch_size=batch_size, collate_fn=data_collator)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = TransformerEncoderModel(num_embeddings=vocab_size, d_model=128, padding_idx=hf_tokenizer.pad_token_id, nhead=8, dim_feedforward=4*128, num_layers=4)
model.load_state_dict(torch.load(model_path))
classifier_model = TransformerClassifierModel(encoder=model, n_classes=n_classes)
classifier_model.to(device)

classifier_model.train()

optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

loss_list = []

loss_val_list = []

# Early stopping variables
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None

for epoch in range(n_epochs):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = classifier_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss_list.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Validation
    classifier_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = classifier_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            loss_val_list.append(loss.item())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Early stopping check
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_model_state = classifier_model.state_dict().copy()
        print(f"New best validation loss: {best_val_loss:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    classifier_model.train()

# Restore best model if early stopping was triggered
if best_model_state is not None:
    classifier_model.load_state_dict(best_model_state)
    print(f"Restored best model with validation loss: {best_val_loss:.4f}")


#save loss history as np array
import numpy as np
np.save("training_loss_char_classification.npy", np.array(loss_list))

#save validation loss history as np array
np.save("validation_loss_char_classification.npy", np.array(loss_val_list))

#save model
classifier_model.save_path = "./models/classifiers/char_model.pth"
torch.save(classifier_model.state_dict(), classifier_model.save_path)


#test model
classifier_model.eval()
correct = 0
total = 0


test_tokenized = dataset_train['test'].map(tokenize, batched=True, batch_size=batch_size, remove_columns=["text"])
test_loader = DataLoader(test_tokenized, batch_size=batch_size, collate_fn=data_collator)
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = classifier_model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f"Test Accuracy: {accuracy*100:.2f}%")