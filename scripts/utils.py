# %%[1] imports
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import kagglehub
from tqdm.notebook import tqdm

# %%[0] download dataset
path = kagglehub.dataset_download("sid321axn/malicious-urls-dataset")
file_path = os.path.join(path, "malicious_phish.csv")
df = pd.read_csv(file_path)

# %%[1] preprocess
df = df[['url', 'type']].dropna()
df['label'] = df['type'].apply(lambda x: 0 if x.lower() == 'benign' else 1)
train_df, test_df = train_test_split(df[['url', 'label']], test_size=0.2, stratify=df['label'], random_state=42)

# %%[2] alphabet and encoding
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%")
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(CHAR2IDX) + 1
MAX_LEN = 200

class URLDataset(Dataset):
    def __init__(self, df):
        self.urls = df['url'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.urls)

    def encode(self, url):
        url = url.lower()[:MAX_LEN]
        return torch.tensor([CHAR2IDX.get(c, 0) for c in url] + [0] * (MAX_LEN - len(url)), dtype=torch.long)

    def __getitem__(self, idx):
        return self.encode(self.urls[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

class CharCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 16, padding_idx=0)
        self.conv1 = nn.Conv1d(16, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)

def train(model, loader, optimizer, criterion):
    model.train()
    for x, y in tqdm(loader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out = model(x) > 0.5
            correct += (out == y).sum().item()
            total += y.size(0)
    print(f"accuracy: {correct / total:.4f}")

# data
train_ds = URLDataset(train_df)
test_ds = URLDataset(test_df)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

# %%[3]raining
model = CharCNN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in tqdm(range(5)):
    train(model, train_dl, optimizer, criterion)
    torch.save(model.state_dict(), f"models/char_cnn_snapshot_{epoch}.pth")
    print(f"\nEpoch {epoch + 1}")
    evaluate(model, test_dl)
# %%
