#!/usr/bin/env python3
"""
Create a visual representation of the CNN architecture used in the PD Signal Classifier.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_network_architecture_diagram():
    """Create a professional network architecture diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Enhanced colors for different layer types
    colors = {
        'input': '#f8f9fa',
        'conv': '#e3f2fd',
        'batch_norm': '#e8f5e8', 
        'pool': '#fff3e0',
        'dropout': '#ffebee',
        'flatten': '#f3e5f5',
        'dense': '#e1f5fe',
        'output': '#e8eaf6'
    }
    
    # Layer specifications with feature map dimensions
    # Format: (x, y, width, height, label, color_key, output_shape)
    layers = [
        # Input layer
        (1, 9, 1.5, 1.5, 'Input\n(100×2×1)', 'input', '100×2×1'),
        
        # Conv Block 1
        (3.5, 9, 1.5, 1.5, 'Conv2D\n16 filters\n3×3, stride=1\nReLU', 'conv', '98×2×16'),
        (5.5, 9, 1.2, 1.5, 'BatchNorm\n2D', 'batch_norm', '98×2×16'),
        (7.2, 9, 1.2, 1.5, 'MaxPool2D\n2×2', 'pool', '49×1×16'),
        (9, 9, 1.2, 1.5, 'Dropout\n20%', 'dropout', '49×1×16'),
        
        # Conv Block 2
        (11, 9, 1.5, 1.5, 'Conv2D\n32 filters\n3×3, stride=1\nReLU', 'conv', '47×1×32'),
        (13, 9, 1.2, 1.5, 'BatchNorm\n2D', 'batch_norm', '47×1×32'),
        (14.7, 9, 1.2, 1.5, 'MaxPool2D\n2×2', 'pool', '23×1×32'),
        (16.3, 9, 1.2, 1.5, 'Dropout\n20%', 'dropout', '23×1×32'),
        
        # Conv Block 3
        (3.5, 6, 1.5, 1.5, 'Conv2D\n64 filters\n3×3, stride=1\nReLU', 'conv', '21×1×64'),
        (5.5, 6, 1.2, 1.5, 'BatchNorm\n2D', 'batch_norm', '21×1×64'),
        (7.2, 6, 1.2, 1.5, 'MaxPool2D\n2×2', 'pool', '10×1×64'),
        (9, 6, 1.2, 1.5, 'Dropout\n20%', 'dropout', '10×1×64'),
        
        # Flatten
        (11.5, 6, 1.5, 1.5, 'Flatten', 'flatten', '640'),
        
        # Dense Block 1
        (3.5, 3, 1.5, 1.5, 'Dense\n128 units\nReLU + L2', 'dense', '128'),
        (5.5, 3, 1.2, 1.5, 'Dropout\n30%', 'dropout', '128'),
        
        # Dense Block 2
        (7.5, 3, 1.5, 1.5, 'Dense\n64 units\nReLU + L2', 'dense', '64'),
        (9.5, 3, 1.2, 1.5, 'Dropout\n20%', 'dropout', '64'),
        
        # Output
        (12, 3, 1.5, 1.5, 'Dense\n2 units\nSoftmax', 'output', '2'),
    ]
    
    # Draw layers with enhanced styling
    boxes = []
    shape_dims = []
    for i, (x, y, w, h, label, color_key, output_shape) in enumerate(layers):
        # Create rounded rectangle with shadow effect
        shadow = FancyBboxPatch(
            (x + 0.05, y - 0.05), w, h,
            boxstyle="round,pad=0.05",
            facecolor='lightgray',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(shadow)
        
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=colors[color_key],
            edgecolor='black',
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(box)
        boxes.append((x + w/2, y + h/2))
        shape_dims.append(output_shape)
        
        # Add main layer text
        ax.text(x + w/2, y + h/2, label, 
                ha='center', va='center', 
                fontsize=9, fontweight='bold', zorder=3)
        
        # Add output shape annotation below each layer
        ax.text(x + w/2, y - 0.3, f'Output: {output_shape}', 
                ha='center', va='center', 
                fontsize=7, style='italic', color='darkblue', zorder=3)
    
    # Enhanced connections with curved arrows
    from matplotlib.patches import ConnectionPatch
    
    # Define connection paths
    connections = [
        # Conv blocks (horizontal flow)
        (0, 1), (1, 2), (2, 3), (3, 4),  # Block 1
        (4, 5), (5, 6), (6, 7), (7, 8),  # Block 2  
        (8, 9), (9, 10), (10, 11), (11, 12),  # Block 3
        # Transition to dense
        (12, 13),  # Flatten to Dense
        # Dense blocks
        (13, 14), (14, 15), (15, 16), (16, 17)  # Dense layers
    ]
    
    for start_idx, end_idx in connections:
        start_x, start_y = boxes[start_idx]
        end_x, end_y = boxes[end_idx]
        
        # Create curved connection
        if abs(start_y - end_y) > 1:  # Vertical connection (between blocks)
            connection = ConnectionPatch(
                (start_x, start_y - 0.75), (end_x, end_y + 0.75),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="darkblue", alpha=0.8,
                connectionstyle="arc3,rad=0.3"
            )
        else:  # Horizontal connection (within block)
            connection = ConnectionPatch(
                (start_x + 0.75, start_y), (end_x - 0.75, end_y),
                "data", "data", 
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="darkblue", alpha=0.8
            )
        ax.add_patch(connection)
    
    # Enhanced title and annotations
    ax.text(9, 11.5, 'PD Signal Classifier - CNN Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    ax.text(9, 11, 'Default Configuration: [16, 32, 64] conv filters, [128, 64] dense units', 
            ha='center', va='center', fontsize=12, style='italic', color='darkblue')
    
    # Add block labels
    ax.text(7, 10.8, 'Convolutional Feature Extraction', 
            ha='center', va='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax.text(7.5, 4.8, 'Classification Head', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.7))
    
    # Add detailed specifications table
    spec_text = """Model Specifications:
• Input: 100×2 Photodiode signal windows
• Kernel size: 3×3 with valid padding  
• Activation: ReLU (hidden), Softmax (output)
• Optimization: Adam with lr=0.001
• Regularization: L2=0.001, Dropout, BatchNorm
• Training: 5-fold CV, Early stopping (patience=10)"""
    
    ax.text(0.5, 5.5, spec_text, 
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    
    # Add parameter count estimation
    param_text = """Parameter Count (approx.):
• Conv layers: ~8,800 params
• Dense layers: ~90,000 params  
• Total: ~100,000 params
• Classes: Conduct vs Keyhole"""
    
    ax.text(14.5, 5.5, param_text,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9))
    
    # Enhanced legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Layer'),
        patches.Patch(color=colors['conv'], label='Convolutional'),
        patches.Patch(color=colors['batch_norm'], label='Batch Normalization'),
        patches.Patch(color=colors['pool'], label='MaxPooling'),
        patches.Patch(color=colors['dropout'], label='Dropout'),
        patches.Patch(color=colors['flatten'], label='Flatten'),
        patches.Patch(color=colors['dense'], label='Dense'),
        patches.Patch(color=colors['output'], label='Output')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    # Add feature map dimension flow
    ax.text(9, 1.5, 'Feature Map Dimensions: 100×2×1 → 98×2×16 → 49×1×16 → 47×1×32 → 23×1×32 → 21×1×64 → 10×1×64 → 640 → 128 → 64 → 2', 
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightsteelblue', alpha=0.8))
    
    # Add loss function note
    ax.text(9, 0.5, 'Loss: Sparse Categorical Crossentropy | Metrics: Accuracy, Precision, Recall, F1-Score', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='darkred')
    
    plt.tight_layout()
    plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Network architecture diagram saved as 'network_architecture.png'")

if __name__ == "__main__":
    create_network_architecture_diagram()