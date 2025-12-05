import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class HybridMLPClassifier(nn.Module):
    def __init__(self, embedding_dim, num_features, feature_hidden_dim=32, dropout=0.3):
        super().__init__()
        
        self.feature_branch = nn.Sequential(
            nn.Linear(num_features, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        combined_dim = embedding_dim + feature_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, embeddings, features):
        feat_out = self.feature_branch(features)
        
        combined = torch.cat([embeddings, feat_out], dim=1)
        
        logits = self.classifier(combined)
        return logits
    

#a


    