"""
Multi-head attention implementation.
This module implements the multi-head attention mechanism as described in the "Attention Is All You Need" paper.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class MultiHeadAttention(BaseAttention):
    """
    Multi-head attention mechanism that allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    Attributes:
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-head attention.
        
        Args:
            query_dim: Dimension of the query vectors
            key_dim: Dimension of the key vectors
            value_dim: Dimension of the value vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        assert value_dim % num_heads == 0, "value_dim must be divisible by num_heads"
        
        super().__init__(
            query_dim=query_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            output_dim=query_dim,  # Output dimension is same as query dimension
            dropout=dropout,
        )
        
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, key_dim)
        self.v_proj = nn.Linear(value_dim, value_dim)
        
        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def _reshape_for_heads(
        self, x: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """
        Reshape tensor for multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _reshape_from_heads(
        self, x: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """
        Reshape tensor from multi-head attention back to original shape.
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Reshaped tensor of shape (batch_size, seq_len, dim)
        """
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, num_queries, query_dim)
            key: Key tensor of shape (batch_size, num_keys, key_dim)
            value: Value tensor of shape (batch_size, num_keys, value_dim)
            mask: Optional mask tensor of shape (batch_size, num_queries, num_keys)
            
        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, num_queries, query_dim)
                - Attention weights of shape (batch_size, num_heads, num_queries, num_keys)
        """
        batch_size = query.size(0)
        num_queries = query.size(1)
        num_keys = key.size(1)
        
        # Project Q, K, V
        q = self.q_proj(query)  # (batch_size, num_queries, query_dim)
        k = self.k_proj(key)  # (batch_size, num_keys, key_dim)
        v = self.v_proj(value)  # (batch_size, num_keys, value_dim)
        
        # Reshape for multi-head attention
        q = self._reshape_for_heads(q, batch_size, num_queries)
        k = self._reshape_for_heads(k, batch_size, num_keys)
        v = self._reshape_for_heads(v, batch_size, num_keys)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape back
        context = self._reshape_from_heads(context, batch_size, num_queries)
        
        # Project output
        output = self.out_proj(context)
        
        return output, attention_weights 