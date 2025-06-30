"""
Author: @twinzies
Date: 2025-06-30
Description: Entry point for training and evaluating the PyTorch topic classifier using data from data_topic/.
"""

import logging

import pandas as pd
import torch
from model.train import evaluate_model, train_model
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configure logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load data

train_df = pd.read_csv("data_topic/train.csv")
test_df = pd.read_csv("data_topic/test.csv")
dev_df = pd.read_csv("data_topic/dev.csv")

logger.info("Data loaded successfully from data_topic/.")

# Combine dev and test for evaluation, or keep separate
eval_df = pd.concat([dev_df, test_df], ignore_index=True)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_eval = eval_df.iloc[:, :-1].values
y_eval = eval_df.iloc[:, -1].values

# Normalize features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

logger.info("Data standardized successfully.")

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_eval_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32)
y_eval_tensor = torch.tensor(y_eval, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(torch.unique(y_train_tensor))
num_epochs = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

logger.info(f"Using device: {device}")

# Train and evaluate

model = train_model(train_loader, input_size, hidden_size, num_classes, num_epochs, device=device)
evaluate_model(model, eval_loader, device=device)
