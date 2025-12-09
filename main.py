"""
Main script to run the LLM evaluation pipeline.

This script:
1. Loads or generates the Q&A dataset
2. Evaluates 5 different prompts on all Q&A pairs
3. Calculates all metrics (accuracy, latency, cost, ROUGE, BLEU, etc.)
4. Generates a comparison dashboard
"""

import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.evaluator import Evaluator
from dashboard.visualizer import create_dashboard
from data.generate_dataset import generate_qa_dataset, save_dataset


def load_or_generate_dataset(dataset_path: str = "data/qa_dataset.json") -> list:
    """
    Load dataset if it exists, otherwise generate it.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        List of Q&A pairs
    """
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Dataset not found. Generating new dataset...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        dataset = generate_qa_dataset(100)
        save_dataset(dataset, dataset_path)
        return dataset


def main():
    """Main execution function."""
    print("=" * 60)
    print("LLM Evaluation Pipeline")
    print("=" * 60)
    print()
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: No API keys found in environment.")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        print()
        print("For demonstration, you can still generate the dataset and")
        print("see the structure, but evaluation will require API keys.")
        print()
    
    # Load or generate dataset
    dataset_path = "data/qa_dataset.json"
    qa_pairs = load_or_generate_dataset(dataset_path)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    print()
    
    # Check if we should run evaluation
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipping evaluation (no API keys).")
        print("To run full evaluation, set API keys in .env file")
        return
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = Evaluator(
        provider="openai",
        model="gpt-3.5-turbo",  # Using cheaper model for cost efficiency
        judge_provider="openai",
        judge_model="gpt-4o-mini"  # Using cheaper judge model
    )
    print("✓ Evaluator initialized")
    print()
    
    # Get all prompt templates
    template_names = evaluator.prompt_templates.get_template_names()
    print(f"Testing {len(template_names)} prompt templates:")
    for name in template_names:
        print(f"  - {name}")
    print()
    
    # Run evaluation
    print("Starting evaluation...")
    print("This may take a while depending on API rate limits.")
    print("Evaluating 5 prompts × 100 Q&A pairs = 500 total evaluations")
    print()
    
    try:
        results = evaluator.evaluate_dataset(qa_pairs, template_names)
        print()
        print("✓ Evaluation complete!")
        print()
        
        # Save results
        results_path = "evaluation_results.json"
        evaluator.save_results(results, results_path)
        print()
        
        # Generate dashboard
        print("Generating dashboard...")
        summary_stats = create_dashboard(results, "evaluation_dashboard.html")
        print()
        
        # Print summary
        print("=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print()
        
        import pandas as pd
        summary_df = pd.DataFrame(summary_stats).T
        summary_df = summary_df.round(4)
        
        print("Average Metrics by Prompt Template:")
        print(summary_df[['avg_accuracy', 'avg_rouge_l', 'avg_bleu', 'avg_latency', 'total_cost']].to_string())
        print()
        
        # Find best template
        best_accuracy = summary_df['avg_accuracy'].idxmax()
        best_latency = summary_df['avg_latency'].idxmin()
        lowest_cost = summary_df['total_cost'].idxmin()
        
        print(f"🏆 Best Accuracy: {best_accuracy} ({summary_df.loc[best_accuracy, 'avg_accuracy']:.4f})")
        print(f"⚡ Fastest: {best_latency} ({summary_df.loc[best_latency, 'avg_latency']:.2f}s)")
        print(f"💰 Lowest Cost: {lowest_cost} (${summary_df.loc[lowest_cost, 'total_cost']:.4f})")
        print()
        print("=" * 60)
        print("Dashboard saved to: evaluation_dashboard.html")
        print("Results saved to: evaluation_results.json")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print("Partial results can be found in evaluation_results.json if saved.")
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

