"""
Main evaluation pipeline that orchestrates testing multiple prompts
on Q&A pairs and collecting metrics.
"""

import os
import time
import json
from typing import Dict, List, Optional
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from .prompt_templates import PromptTemplates
from ..metrics.rouge_bleu import calculate_rouge, calculate_bleu
from ..metrics.llm_judge import LLMJudge
from ..metrics.custom_metrics import (
    calculate_length_ratio,
    calculate_keyword_overlap,
    detect_hallucination_indicators
)

load_dotenv()


class Evaluator:
    """
    Main evaluator that tests multiple prompts on Q&A pairs.
    
    For each prompt template and Q&A pair, it:
    1. Generates LLM response
    2. Measures latency
    3. Estimates cost
    4. Calculates all metrics
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        judge_provider: str = "openai",
        judge_model: str = "gpt-4o-mini"
    ):
        """
        Initialize the evaluator.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model to use for generating responses
            judge_provider: Provider for judge LLM
            judge_model: Model to use as judge
        """
        self.provider = provider
        self.model = model
        self.prompt_templates = PromptTemplates()
        
        # Initialize LLM client
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
            # Cost per 1K tokens (approximate, varies by model)
            self.cost_per_1k_tokens = {
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            }.get(model, {"input": 0.0015, "output": 0.002})
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = Anthropic(api_key=api_key)
            self.cost_per_1k_tokens = {
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            }.get(model, {"input": 0.00025, "output": 0.00125})
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Initialize judge
        self.judge = LLMJudge(provider=judge_provider, model=judge_model)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate API cost based on token usage."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_tokens["input"]
        output_cost = (output_tokens / 1000) * self.cost_per_1k_tokens["output"]
        return input_cost + output_cost
    
    def get_llm_response(
        self,
        prompt: str,
        max_tokens: int = 500
    ) -> tuple[str, float, float]:
        """
        Get response from LLM with latency and cost tracking.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (response_text, latency_seconds, estimated_cost)
        """
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                response_text = completion.choices[0].message.content
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
            else:  # anthropic
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = message.content[0].text
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
            
            latency = time.time() - start_time
            cost = self._estimate_cost(input_tokens, output_tokens)
            
            return response_text, latency, cost
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return f"Error: {str(e)}", time.time() - start_time, 0.0
    
    def evaluate_single(
        self,
        question: str,
        reference: str,
        prompt_template_name: str
    ) -> Dict:
        """
        Evaluate a single Q&A pair with a specific prompt template.
        
        Args:
            question: The question
            reference: Reference answer
            prompt_template_name: Name of prompt template to use
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Get prompt template
        templates = self.prompt_templates.get_all_templates()
        if prompt_template_name not in templates:
            raise ValueError(f"Unknown template: {prompt_template_name}")
        
        prompt = templates[prompt_template_name](question)
        
        # Get LLM response
        response, latency, cost = self.get_llm_response(prompt)
        
        # Calculate metrics
        rouge_scores = calculate_rouge(response, reference)
        bleu_scores = calculate_bleu(response, reference)
        
        # LLM judge evaluation
        judge_scores = self.judge.evaluate(question, response, reference)
        
        # Custom metrics
        length_ratio = calculate_length_ratio(response, reference)
        keyword_overlap = calculate_keyword_overlap(response, reference)
        hallucination_indicators = detect_hallucination_indicators(response, reference)
        
        return {
            'question': question,
            'reference': reference,
            'response': response,
            'prompt_template': prompt_template_name,
            'latency': latency,
            'cost': cost,
            'metrics': {
                'rouge': rouge_scores,
                'bleu': bleu_scores['bleu'],
                'judge': judge_scores,
                'length_ratio': length_ratio,
                'keyword_overlap': keyword_overlap,
                'hallucination_risk': hallucination_indicators['hallucination_risk']
            }
        }
    
    def evaluate_dataset(
        self,
        qa_pairs: List[Dict[str, str]],
        prompt_template_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Evaluate all Q&A pairs with all prompt templates.
        
        Args:
            qa_pairs: List of dicts with 'question' and 'reference' keys
            prompt_template_names: Optional list of template names to test
            
        Returns:
            List of evaluation results
        """
        if prompt_template_names is None:
            prompt_template_names = self.prompt_templates.get_template_names()
        
        results = []
        total_evaluations = len(qa_pairs) * len(prompt_template_names)
        
        with tqdm(total=total_evaluations, desc="Evaluating") as pbar:
            for qa_pair in qa_pairs:
                question = qa_pair['question']
                reference = qa_pair['reference']
                
                for template_name in prompt_template_names:
                    result = self.evaluate_single(
                        question, reference, template_name
                    )
                    results.append(result)
                    pbar.update(1)
        
        return results
    
    def save_results(self, results: List[Dict], filepath: str):
        """Save evaluation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")

