"""
Training utilities for GPT.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class GPTTrainer:
    """Trainer for GPT models."""
    
    def __init__(self, model, train_loader, val_loader=None, optimizer=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=3e-4)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(tqdm(self.train_loader)):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
