#!/usr/bin/env python3
"""
DoE Workflow Integration Example

This script demonstrates how to use the integrated DoE system with your training loop.
Shows complete workflow from experiment generation through execution to analysis.
"""

import subprocess
import sys
from pathlib import Path

def run_doe_workflow(classifier_type='cwt_image', verbose=True):
    """
    Complete DoE workflow example showing all phases.
    
    Args:
        classifier_type: 'pd_signal' or 'cwt_image'  
        verbose: Show detailed output
    """
    
    print("🧪 DoE INTEGRATED TRAINING WORKFLOW")
    print("=" * 50)
    
    # Phase 1: Screening Design (Factorial)
    print("\n📋 PHASE 1: SCREENING DESIGN")
    print("-" * 30)
    
    cmd_phase1 = [
        'python', 'ml/hyperparameter_tuner.py',
        '--classifier', classifier_type,
        '--mode', 'doe',
        '--doe_design', 'factorial',
        '--doe_phase', '1',
        '--auto_analyze'
    ]
    
    if verbose:
        cmd_phase1.append('--verbose')
    
    print("Command:", ' '.join(cmd_phase1))
    print("Expected: ~32-44 experiments, 2-3 days runtime")
    print("Goal: Identify main effects and interactions")
    
    # Uncomment to actually run:
    # subprocess.run(cmd_phase1)
    
    # Phase 2: Response Surface Methodology 
    print("\n📈 PHASE 2: RESPONSE SURFACE DESIGN")  
    print("-" * 35)
    
    # This would typically use factors identified from Phase 1 analysis
    top_factors = ['learning_rate', 'batch_size', 'conv_dropout']  # Example from Phase 1 results
    
    cmd_phase2 = [
        'python', 'ml/hyperparameter_tuner.py', 
        '--classifier', classifier_type,
        '--mode', 'doe',
        '--doe_design', 'response_surface',
        '--doe_phase', '2',
        '--doe_factors'] + top_factors + [
        '--auto_analyze'
    ]
    
    if verbose:
        cmd_phase2.append('--verbose')
    
    print("Command:", ' '.join(cmd_phase2))
    print("Expected: ~40-60 experiments, 3-4 days runtime")
    print("Goal: Optimize around promising regions")
    
    # Phase 3: Validation experiments
    print("\n✅ PHASE 3: VALIDATION")
    print("-" * 20)
    
    cmd_phase3 = [
        'python', 'ml/hyperparameter_tuner.py',
        '--classifier', classifier_type, 
        '--mode', 'doe',
        '--doe_design', 'lhs',
        '--doe_phase', '3',
        '--auto_analyze'
    ]
    
    if verbose:
        cmd_phase3.append('--verbose')
        
    print("Command:", ' '.join(cmd_phase3))
    print("Expected: ~20-25 experiments, 1-2 days runtime")
    print("Goal: Confirm optimal settings and robustness")
    
    print("\n🎯 ALTERNATIVE: SINGLE-PHASE APPROACHES")
    print("-" * 45)
    
    # Alternative 1: Load pre-generated experiments
    print("\n1. Load from pre-generated CSV:")
    print("   python ml/generate_doe_experiments.py --mode cwt --design factorial --phase 1")
    print("   python ml/hyperparameter_tuner.py --mode doe --doe_design from_file --doe_file path/to/experiments.csv")
    
    # Alternative 2: Quick exploratory design
    print("\n2. Quick exploratory design (LHS):")
    print("   python ml/hyperparameter_tuner.py --classifier cwt_image --mode doe --doe_design lhs --auto_analyze")
    
    # Alternative 3: Focus on specific parameters
    print("\n3. Focus on specific parameters:")
    print("   python ml/hyperparameter_tuner.py --classifier cwt_image --mode doe --doe_design factorial \\")
    print("      --doe_factors learning_rate batch_size conv_dropout --auto_analyze")
    
    print("\n📊 ANALYSIS INTEGRATION")
    print("-" * 25)
    print("- Automatic analysis after each phase (--auto_analyze)")
    print("- Interaction effects detection")
    print("- Parameter importance ranking") 
    print("- DoE recommendations for next phase")
    print("- Statistical validation (ANOVA, power analysis)")
    
    print("\n🎮 TRY IT NOW")
    print("-" * 15)
    print("Quick test with minimal experiments:")
    print("python ml/hyperparameter_tuner.py --classifier cwt_image --mode doe \\")
    print("  --doe_design factorial --doe_factors learning_rate batch_size \\")
    print("  --max_configs 8 --auto_analyze --verbose")


def demonstrate_integration_features():
    """Show the key integration features."""
    
    print("\n🔗 KEY INTEGRATION FEATURES")
    print("=" * 35)
    
    features = [
        "✅ Direct integration with existing hyperparameter tuner",
        "✅ No config file management needed",
        "✅ Automatic experiment execution",
        "✅ Built-in deduplication (reuses previous results)",
        "✅ Progress tracking and resume capability", 
        "✅ Automatic comprehensive analysis",
        "✅ Statistical validation and reporting",
        "✅ DoE recommendations for next experiments",
        "✅ Support for all classifier types (PD/CWT)",
        "✅ Interaction effects detection"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n📈 STATISTICAL BENEFITS")
    print("-" * 25)
    
    benefits = [
        "• Systematic parameter space exploration",
        "• Statistical significance testing", 
        "• Interaction effects quantification",
        "• Optimal resource allocation",
        "• Balanced experimental design",
        "• Power analysis for adequate sampling",
        "• Model adequacy diagnostics",
        "• Prediction intervals for new configs"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DoE Workflow Integration Example')
    parser.add_argument('--classifier', choices=['pd_signal', 'cwt_image'], default='cwt_image',
                       help='Classifier type for demonstration')
    parser.add_argument('--demo_only', action='store_true', 
                       help='Show workflow commands without executing')
    
    args = parser.parse_args()
    
    if args.demo_only:
        demonstrate_integration_features()
    else:
        run_doe_workflow(args.classifier, verbose=True)
        demonstrate_integration_features()