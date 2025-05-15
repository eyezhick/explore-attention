"""
Visualization utilities for attention mechanisms.
This module provides functions for visualizing attention weights and other related plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from typing import List, Optional, Tuple, Union


def plot_attention_weights(
    attention_weights: Union[torch.Tensor, np.ndarray],
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor/array of shape (num_queries, num_keys)
        x_labels: Labels for the x-axis (keys)
        y_labels: Labels for the y-axis (queries)
        title: Title of the plot
        figsize: Figure size as (width, height)
        cmap: Colormap for the heatmap
        save_path: Optional path to save the plot
    """
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
    )
    
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()


def plot_attention_heads(
    attention_weights: torch.Tensor,
    num_heads: int,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Multi-Head Attention Weights",
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot attention weights for multiple heads in a grid.
    
    Args:
        attention_weights: Attention weights tensor of shape (num_heads, num_queries, num_keys)
        num_heads: Number of attention heads
        x_labels: Labels for the x-axis (keys)
        y_labels: Labels for the y-axis (queries)
        title: Title of the plot
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
    """
    # Convert to numpy
    attention_weights = attention_weights.detach().cpu().numpy()
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_heads)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each head
    for i in range(num_heads):
        sns.heatmap(
            attention_weights[i],
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap="viridis",
            ax=axes[i],
            cbar=False,
        )
        axes[i].set_title(f"Head {i+1}")
    
    # Remove empty subplots
    for i in range(num_heads, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()


def plot_attention_flow(
    attention_weights: torch.Tensor,
    tokens: List[str],
    title: str = "Attention Flow",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot attention flow between tokens using a directed graph.
    
    Args:
        attention_weights: Attention weights tensor of shape (num_queries, num_keys)
        tokens: List of token strings
        title: Title of the plot
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
    """
    import networkx as nx
    
    # Convert to numpy
    attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, token in enumerate(tokens):
        G.add_node(i, label=token)
    
    # Add edges with weights
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if attention_weights[i, j] > 0.1:  # Only show significant connections
                G.add_edge(i, j, weight=attention_weights[i, j])
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos, {i: token for i, token in enumerate(tokens)})
    
    plt.title(title)
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()


def plot_attention_evolution(
    attention_weights: torch.Tensor,
    layer_names: List[str],
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Attention Evolution Across Layers",
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot how attention weights evolve across different layers.
    
    Args:
        attention_weights: Attention weights tensor of shape (num_layers, num_queries, num_keys)
        layer_names: Names of the layers
        x_labels: Labels for the x-axis (keys)
        y_labels: Labels for the y-axis (queries)
        title: Title of the plot
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
    """
    # Convert to numpy
    attention_weights = attention_weights.detach().cpu().numpy()
    num_layers = len(layer_names)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_layers)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each layer
    for i in range(num_layers):
        sns.heatmap(
            attention_weights[i],
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap="viridis",
            ax=axes[i],
            cbar=False,
        )
        axes[i].set_title(f"Layer: {layer_names[i]}")
    
    # Remove empty subplots
    for i in range(num_layers, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show() 