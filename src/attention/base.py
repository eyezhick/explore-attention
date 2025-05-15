"""
Base attention mechanism implementation.
This module provides the foundation for all attention mechanisms in the project.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):
    """
    Base attention mechanism that implements the core attention computation.
    
    This class provides the fundamental attention mechanism that can be extended
    to implement various types of attention (e.g., additive, multiplicative, etc.).
    
    Attributes:
        query_dim (int): Dimension of the query vectors
        key_dim (int): Dimension of the key vectors
        value_dim (int): Dimension of the value vectors
        output_dim (int): Dimension of the output vectors
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize the base attention mechanism.
        
        Args:
            query_dim: Dimension of the query vectors
            key_dim: Dimension of the key vectors
            value_dim: Dimension of the value vectors
            output_dim: Dimension of the output vectors (defaults to value_dim)
            dropout: Dropout probability
        """
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim or value_dim
        
        # Linear projections
        self.query_proj = nn.Linear(query_dim, self.output_dim)
        self.key_proj = nn.Linear(key_dim, self.output_dim)
        self.value_proj = nn.Linear(value_dim, self.output_dim)
        self.output_proj = nn.Linear(self.output_dim, self.output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for dot-product attention
        self.scale = math.sqrt(self.output_dim)

    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention weights using dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, num_queries, query_dim)
            key: Key tensor of shape (batch_size, num_keys, key_dim)
            mask: Optional mask tensor of shape (batch_size, num_queries, num_keys)
            
        Returns:
            Attention weights of shape (batch_size, num_queries, num_keys)
        """
        # Project queries and keys
        query = self.query_proj(query)  # (batch_size, num_queries, output_dim)
        key = self.key_proj(key)  # (batch_size, num_keys, output_dim)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention mechanism.
        
        Args:
            query: Query tensor of shape (batch_size, num_queries, query_dim)
            key: Key tensor of shape (batch_size, num_keys, key_dim)
            value: Value tensor of shape (batch_size, num_keys, value_dim)
            mask: Optional mask tensor of shape (batch_size, num_queries, num_keys)
            
        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, num_queries, output_dim)
                - Attention weights of shape (batch_size, num_queries, num_keys)
        """
        # Project values
        value = self.value_proj(value)  # (batch_size, num_keys, output_dim)
        
        # Compute attention weights
        attention_weights = self.compute_attention_weights(query, key, mask)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, value)
        
        # Project output
        output = self.output_proj(context)
        
        return output, attention_weights 