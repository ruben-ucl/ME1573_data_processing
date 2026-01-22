#!/usr/bin/env python3
"""
Create a data flow diagram for the PD Signal Classifier pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_dataflow_diagram():
    """Create a comprehensive data flow diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Enhanced colors for different process types
    colors = {
        'data': '#e8f5e8',
        'preprocessing': '#fff3e0', 
        'training': '#e3f2fd',
        'validation': '#f3e5f5',
        'output': '#ffebee',
        'decision': '#e1f5fe',
        'hyperopt': '#f0f4ff',
        'storage': '#fafafa'
    }
    
    # Enhanced data flow components with detailed specifications
    # Format: (x, y, width, height, label, color_key, shape, details)
    components = [
        # Data Sources & Input
        (1, 12, 3, 1.2, 'Raw Data Sources\n16-bit TIFF Images\n~2000 samples\nConduct/Keyhole Classes', 'data', 'rect', 'Input: F:/AlSi10Mg/.../window_plots_16bit/'),
        
        # Hyperparameter Optimization Control
        (16, 12, 3, 1.2, 'Hyperparameter Tuner\nGrid Search Control\nConfig Generation\nProgress Tracking', 'hyperopt', 'rect', 'Quick/Smart/Full modes'),
        
        # Preprocessing Pipeline (Row 1)
        (1, 10, 2.5, 1, 'Data Loading\n16-bit → float32\nNormalize [0,1]\nResize (100×2)', 'preprocessing', 'rect', 'cv2.imread + normalization'),
        (4.5, 10, 2.5, 1, 'Data Validation\nClass Distribution\nPixel Range Check\nCorruption Detection', 'preprocessing', 'rect', 'Quality assurance'),
        (8, 10, 2.5, 1, 'Train/Val Split\nStratified K-Fold\n80% Train, 20% Val\nk=5 folds', 'preprocessing', 'rect', 'sklearn.StratifiedKFold'),
        
        # Training Control Loop
        (12, 10, 2.5, 1, 'Fold Iteration\nLoop k=1 to 5\nIndependent Models\nProgress Tracking', 'decision', 'diamond', 'Cross-validation controller'),
        
        # Training Pipeline (Row 2)
        (1, 7.5, 2.5, 1, 'Data Augmentation\nTime Shift ±5%\nGaussian Noise\nAmplitude Scaling', 'training', 'rect', 'Optional: 50% probability'),
        (4.5, 7.5, 2.5, 1, 'Batch Processing\nClass Weighting\nShuffle & Batch\nMemory Management', 'training', 'rect', 'tf.data.Dataset pipeline'),
        (8, 7.5, 2.5, 1, 'CNN Forward Pass\nFeature Extraction\nClassification Head\nLoss Calculation', 'training', 'rect', 'Model architecture'),
        
        # Validation & Control (Row 2)
        (12, 7.5, 2.5, 1, 'Epoch Validation\nMetrics Calculation\nAccuracy, F1-Score\nConfusion Matrix', 'validation', 'rect', 'Per-epoch evaluation'),
        (15.5, 7.5, 2.5, 1, 'Training Control\nEarly Stopping\nLR Reduction\nCheckpointing', 'decision', 'diamond', 'Callbacks & optimization'),
        
        # Backward Pass & Optimization
        (8, 5.5, 2.5, 1, 'Backward Pass\nGradient Calculation\nAdam Optimizer\nWeight Updates', 'training', 'rect', 'Backpropagation'),
        
        # Results Collection (Row 3)
        (1, 4, 2.5, 1, 'Fold Completion\nValidation Metrics\nModel Checkpoint\nTiming Info', 'output', 'rect', 'Per-fold results'),
        (4.5, 4, 2.5, 1, 'Model Artifacts\nTrained Weights\nTraining History\nArchitecture Config', 'storage', 'rect', 'HDF5 + JSON files'),
        (8, 4, 2.5, 1, 'Experiment Logging\nUnified CSV Log\nHyperparameters\nPerformance Metrics', 'output', 'rect', 'logs/experiment_log.csv'),
        
        # Cross-Validation Aggregation
        (12, 4, 3, 1, 'CV Aggregation\nMean ± Std Accuracy\nBest Fold Selection\nStatistical Analysis', 'validation', 'rect', 'Cross-validation summary'),
        
        # Final Results & Analysis
        (4, 1.5, 8, 1.2, 'Final Results & Analysis\nMean Validation Accuracy ± Standard Deviation\nBest Fold Performance, Training Time, Convergence Status\nModel Selection & Hyperparameter Recommendations', 'output', 'rect', 'Complete experiment summary'),
        
        # Hyperparameter Optimization Results
        (13, 1.5, 6, 1.2, 'Hyperparameter Optimization Results\nBest Configuration Selection\nPerformance Comparison\nOptimization Recommendations', 'hyperopt', 'rect', 'hyperopt_results/ directory'),
    ]
    
    # Draw components with enhanced styling
    boxes = []
    for i, (x, y, w, h, label, color_key, shape, details) in enumerate(components):
        # Add shadow effect
        shadow = FancyBboxPatch(
            (x + 0.08, y - 0.08), w, h,
            boxstyle="round,pad=0.05",
            facecolor='lightgray',
            alpha=0.4,
            zorder=1
        )
        ax.add_patch(shadow)
        
        if shape == 'diamond':
            # Create diamond shape for decision points
            diamond = patches.RegularPolygon((x + w/2, y + h/2), 4, radius=0.8, 
                                           orientation=np.pi/4,
                                           facecolor=colors[color_key],
                                           edgecolor='black', linewidth=2,
                                           zorder=2)
            ax.add_patch(diamond)
        else:
            # Create rounded rectangle with enhanced styling
            box = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.1",
                facecolor=colors[color_key],
                edgecolor='black',
                linewidth=1.8,
                zorder=2
            )
            ax.add_patch(box)
        
        boxes.append((x + w/2, y + h/2))
        
        # Add main label text
        ax.text(x + w/2, y + h/2, label, 
                ha='center', va='center', 
                fontsize=8.5, fontweight='bold', zorder=3)
        
        # Add technical details below each component
        ax.text(x + w/2, y - 0.25, details, 
                ha='center', va='center', 
                fontsize=6.5, style='italic', color='darkblue', zorder=3)
    
    # Enhanced data flow arrows with clear pathways
    from matplotlib.patches import ConnectionPatch
    
    # Define main data flow paths
    main_flows = [
        # Data preprocessing pipeline (horizontal)
        (0, 2), (2, 3), (3, 4),  # Raw data → Loading → Validation → Splitting
        
        # Hyperparameter control
        (1, 4),  # Hyperparameter tuner → Fold iteration
        
        # Training pipeline (horizontal)
        (5, 6), (6, 7),  # Augmentation → Batch processing → CNN forward
        
        # Validation flow
        (7, 8), (8, 9),  # CNN → Validation → Training control
        
        # Backward pass
        (7, 10),  # CNN forward → Backward pass
        
        # Results collection
        (8, 11), (11, 12), (12, 13),  # Validation → Fold results → Artifacts → Logging
        
        # Cross-validation aggregation
        (11, 14),  # Fold results → CV aggregation
        
        # Final outputs
        (14, 15), (14, 16),  # CV aggregation → Final results and Hyperopt results
    ]
    
    # Feedback and control flows
    feedback_flows = [
        (4, 5),   # Fold iteration → Augmentation (fold loop start)
        (9, 10),  # Training control → Backward pass (continue training)
        (10, 7),  # Backward pass → CNN forward (next epoch)
        (4, 14),  # Fold iteration → CV aggregation (next fold)
    ]
    
    # Draw main flows
    for start_idx, end_idx in main_flows:
        start_x, start_y = boxes[start_idx]
        end_x, end_y = boxes[end_idx]
        
        connection = ConnectionPatch(
            (start_x + 1.25, start_y), (end_x - 1.25, end_y),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc="darkblue", ec="darkblue", alpha=0.8,
            linewidth=2.5
        )
        ax.add_patch(connection)
    
    # Draw feedback flows with different styling
    for start_idx, end_idx in feedback_flows:
        start_x, start_y = boxes[start_idx]
        end_x, end_y = boxes[end_idx]
        
        connection = ConnectionPatch(
            (start_x, start_y - 0.5), (end_x, end_y + 0.5),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=18, fc="red", ec="red", alpha=0.7,
            linewidth=2, linestyle='dashed',
            connectionstyle="arc3,rad=0.3"
        )
        ax.add_patch(connection)
    
    # Enhanced title and section labels
    ax.text(10, 13.5, 'PD Signal Classifier - Complete Data Flow Pipeline', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax.text(10, 13, 'End-to-End Machine Learning Workflow with Hyperparameter Optimization', 
            ha='center', va='center', fontsize=14, style='italic', color='darkblue')
    
    # Add workflow stage labels
    ax.text(5.5, 11.2, 'Data Preprocessing Pipeline', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    ax.text(9, 8.7, 'Training & Validation Loop', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    ax.text(8, 3.2, 'Results Collection & Analysis', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    # Add flow type annotations
    ax.text(1, 6, 'Main Data Flow', ha='left', va='center', 
            fontsize=10, color='darkblue', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    ax.text(18, 6, 'Control & Feedback', ha='right', va='center', 
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    # Enhanced legend with more categories
    legend_elements = [
        patches.Patch(color=colors['data'], label='Data Sources'),
        patches.Patch(color=colors['preprocessing'], label='Preprocessing'),
        patches.Patch(color=colors['training'], label='Training'),
        patches.Patch(color=colors['validation'], label='Validation'),
        patches.Patch(color=colors['decision'], label='Control Flow'),
        patches.Patch(color=colors['output'], label='Results/Logging'),
        patches.Patch(color=colors['hyperopt'], label='Hyperparameter Optimization'),
        patches.Patch(color=colors['storage'], label='Model Storage')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.95), fontsize=10)
    
    # Add comprehensive technical specifications
    tech_specs = """Technical Specifications:
• Input: 16-bit TIFF images (100×2 pixels)
• Model: CNN with 3 conv blocks + 2 dense layers
• Training: 5-fold stratified cross-validation
• Optimization: Adam (lr=0.001, β₁=0.9, β₂=0.999)
• Regularization: L2=0.001, Dropout, BatchNorm
• Loss: Sparse categorical crossentropy
• Metrics: Accuracy, Precision, Recall, F1-Score
• Early stopping: Patience=10 epochs
• Hyperparameters: Learning rate, batch size, architecture"""
    
    ax.text(0.5, 2.5, tech_specs, 
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcyan', alpha=0.9))
    
    # Add file structure and outputs
    outputs_info = """Output Structure:
• logs/experiment_log.csv - Unified experiment tracking
• outputs/v###/ - Versioned experiment folders
• hyperopt_results/ - Optimization results & configs
• Models: HDF5 format with training history
• Plots: Confusion matrices, training curves
• Progress: Real-time ETA and performance tracking
• Resume: Checkpoint support for long runs"""
    
    ax.text(14, 2.5, outputs_info,
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightsteelblue', alpha=0.9))
    
    # Add performance metrics footer
    ax.text(10, 0.3, 'Performance Tracking: Real-time progress monitoring • ETA calculation • Concise/verbose output modes • Experiment versioning • Result comparison', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('dataflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Data flow diagram saved as 'dataflow_diagram.png'")

if __name__ == "__main__":
    create_dataflow_diagram()