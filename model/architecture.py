"""
Description: A simple PyTorch architecture for topic classification.
"""

import torch.nn as nn


class TopicClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TopicClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
