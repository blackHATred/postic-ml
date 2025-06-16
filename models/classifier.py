"""ML модели."""
import torch
import torch.nn as nn


class TwoLayerClassifier(nn.Module):
    """Двухслойный классификатор для определения необходимости поддержки."""
    
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
