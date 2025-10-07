#!/usr/bin/env python3
"""
Channel Analysis Script for Multi-Channel CWT Analysis

This script provides functions for analyzing channel contributions and interactions
in multi-channel CWT experiments. It can be used as a standalone script or imported
as a module for programmatic analysis.

Usage:
    python ml/channel_analysis.py --results results.csv --study "flow_analysis"
    
    # Or as a module:
    from channel_analysis import analyze_channel_contributions
"""

import os
import pandas as pd
import argparse
from pathlib import Path
from itertools import combinations
import json

# Set UTF-8 encoding for Windows compatibility
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

def analyze_channel_contributions(ablation_results):
    """
    Analyze channel contributions and interactions from ablation study results.
    
    Args:
        ablation_results (list): List of dictionaries containing ablation results.
                                Each dict should have: ablation_study, ablation_channels, mean_val_accuracy
    
    Returns:
        dict: Comprehensive analysis including individual contributions, interaction effects, summary stats
    """
    if not ablation_results:
        raise ValueError("No ablation results provided")
    
    # Organize results by channel configuration
    results_by_config = {}
    all_channels = set()
    
    for result in ablation_results:
        channels = result['ablation_channels']
        if isinstance(channels, str):
            channels = [ch.strip() for ch in channels.split(',')]
        
        channels_key = tuple(sorted(channels))
        results_by_config[channels_key] = result
        all_channels.update(channels)
    
    all_channels = sorted(list(all_channels))
    n_channels = len(all_channels)
    
    # Find baseline (all channels) performance
    baseline_key = tuple(sorted(all_channels))
    if baseline_key not in results_by_config:
        raise ValueError(f"Baseline configuration with all channels {all_channels} not found in results")
    
    baseline_accuracy = results_by_config[baseline_key]['mean_val_accuracy']
    
    # Calculate individual channel contributions
    individual_contributions = {}
    for channel in all_channels:
        channel_key = (channel,)
        if channel_key in results_by_config:
            individual_acc = results_by_config[channel_key]['mean_val_accuracy']
            relative_contribution = ((individual_acc - (1/2)) / (baseline_accuracy - (1/2))) * 100  # Relative to random performance
            
            individual_contributions[channel] = {
                'individual_accuracy': individual_acc,
                'baseline_accuracy': baseline_accuracy,
                'relative_contribution': relative_contribution,
                'absolute_difference': individual_acc - baseline_accuracy
            }
        else:
            individual_contributions[channel] = {
                'individual_accuracy': None,
                'baseline_accuracy': baseline_accuracy,
                'relative_contribution': None,
                'absolute_difference': None
            }
    
    # Calculate pairwise interaction effects
    interaction_effects = {}
    for pair in combinations(all_channels, 2):
        pair_key = tuple(sorted(pair))
        if pair_key in results_by_config:
            pair_accuracy = results_by_config[pair_key]['mean_val_accuracy']
            
            # Get individual accuracies
            ind1_acc = individual_contributions[pair[0]]['individual_accuracy']
            ind2_acc = individual_contributions[pair[1]]['individual_accuracy']
            
            if ind1_acc is not None and ind2_acc is not None:
                # Calculate expected additive accuracy
                expected_additive = (ind1_acc + ind2_acc) / 2
                
                # Calculate synergy factor
                synergy_factor = pair_accuracy / expected_additive if expected_additive > 0 else 0
                
                interaction_effects[pair] = {
                    'pair_accuracy': pair_accuracy,
                    'individual_1_accuracy': ind1_acc,
                    'individual_2_accuracy': ind2_acc,
                    'expected_additive': expected_additive,
                    'synergy_factor': synergy_factor,
                    'interaction_effect': pair_accuracy - expected_additive
                }
    
    # Calculate summary statistics
    individual_accs = [contrib['individual_accuracy'] for contrib in individual_contributions.values() 
                      if contrib['individual_accuracy'] is not None]
    
    best_individual_channel = None
    best_individual_accuracy = 0
    for channel, contrib in individual_contributions.items():
        if contrib['individual_accuracy'] is not None and contrib['individual_accuracy'] > best_individual_accuracy:
            best_individual_accuracy = contrib['individual_accuracy']
            best_individual_channel = channel
    
    multi_channel_benefit = baseline_accuracy - best_individual_accuracy if best_individual_accuracy > 0 else 0
    multi_channel_benefit_percent = (multi_channel_benefit / best_individual_accuracy * 100) if best_individual_accuracy > 0 else 0
    
    summary = {
        'baseline_accuracy': baseline_accuracy,
        'best_individual_channel': best_individual_channel,
        'best_individual_accuracy': best_individual_accuracy,
        'multi_channel_benefit': multi_channel_benefit,
        'multi_channel_benefit_percent': multi_channel_benefit_percent,
        'mean_individual_accuracy': sum(individual_accs) / len(individual_accs) if individual_accs else 0,
        'individual_accuracy_std': pd.Series(individual_accs).std() if len(individual_accs) > 1 else 0,
        'number_of_channels': n_channels,
        'channels_tested': all_channels
    }
    
    return {
        'individual_contributions': individual_contributions,
        'interaction_effects': interaction_effects,
        'summary': summary,
        'raw_results': results_by_config
    }

def load_ablation_results_from_csv(csv_path, study_name_filter=None):
    """
    Load ablation results from experiment log CSV.
    
    Args:
        csv_path (str): Path to experiment log CSV
        study_name_filter (str, optional): Filter results by ablation_study name
    
    Returns:
        list: List of ablation result dictionaries
    """
    df = pd.read_csv(csv_path)
    
    # Filter for ablation studies
    ablation_df = df[df['ablation_study'].notna()].copy()
    
    if study_name_filter:
        ablation_df = ablation_df[ablation_df['ablation_study'].str.contains(study_name_filter, na=False)]
    
    if ablation_df.empty:
        raise ValueError(f"No ablation results found{' for study: ' + study_name_filter if study_name_filter else ''}")
    
    # Convert to list of dictionaries
    results = []
    for _, row in ablation_df.iterrows():
        results.append({
            'ablation_study': row['ablation_study'],
            'ablation_channels': row['ablation_channels'],
            'mean_val_accuracy': row['mean_val_accuracy']
        })
    
    return results

def print_analysis_report(analysis):
    """Print a formatted analysis report to console."""
    print("\n" + "="*80)
    print("CHANNEL CONTRIBUTION ANALYSIS REPORT")
    print("="*80)
    
    summary = analysis['summary']
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Baseline (all channels): {summary['baseline_accuracy']:.4f}")
    print(f"  Best individual channel: {summary['best_individual_channel']} ({summary['best_individual_accuracy']:.4f})")
    print(f"  Multi-channel benefit: +{summary['multi_channel_benefit_percent']:.1f}% ({summary['multi_channel_benefit']:+.4f})")
    print(f"  Number of channels: {summary['number_of_channels']}")
    print(f"  Channels tested: {', '.join(summary['channels_tested'])}")
    
    print(f"\nINDIVIDUAL CHANNEL CONTRIBUTIONS:")
    for channel, contrib in analysis['individual_contributions'].items():
        if contrib['individual_accuracy'] is not None:
            print(f"  {channel:15s}: {contrib['individual_accuracy']:.4f} "
                  f"({contrib['relative_contribution']:+6.1f}% relative contribution)")
        else:
            print(f"  {channel:15s}: No individual results available")
    
    if analysis['interaction_effects']:
        print(f"\nPAIRWISE INTERACTION EFFECTS:")
        for pair, effect in analysis['interaction_effects'].items():
            pair_name = ' + '.join(pair)
            synergy = effect['synergy_factor']
            interaction = effect['interaction_effect']
            print(f"  {pair_name:25s}: synergy factor = {synergy:.3f}x, "
                  f"interaction effect = {interaction:+.4f}")
    
    print("\n" + "="*80)

def save_analysis_report(analysis, output_path):
    """Save analysis report to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"Analysis report saved to: {output_path}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze multi-channel ablation study results')
    parser.add_argument('--results', required=True, help='Path to experiment log CSV file')
    parser.add_argument('--study', help='Filter by ablation study name (partial match)')
    parser.add_argument('--output', help='Save analysis to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Load results
        if args.verbose:
            print(f"Loading ablation results from: {args.results}")
        
        results = load_ablation_results_from_csv(args.results, args.study)
        
        if args.verbose:
            print(f"Found {len(results)} ablation results")
        
        # Analyze contributions
        analysis = analyze_channel_contributions(results)
        
        # Print report
        print_analysis_report(analysis)
        
        # Save to file if requested
        if args.output:
            save_analysis_report(analysis, args.output)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())