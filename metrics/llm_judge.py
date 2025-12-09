"""
LLM-as-Judge implementation for evaluating LLM outputs.

This uses a judge LLM to evaluate the quality of another LLM's response.
The judge LLM can understand context, nuance, and semantic meaning
beyond simple n-gram matching.
"""

import os
import time
from typing import Dict, Optional
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMJudge:
    """
    Uses an LLM as a judge to evaluate response quality.
    
    The judge evaluates:
    - Accuracy: Is the information correct?
    - Relevance: Does it answer the question?
    - Completeness: Is the answer complete?
    - Hallucination: Are there made-up facts?
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Initialize the LLM judge.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name to use as judge
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _create_judge_prompt(
        self,
        question: str,
        response: str,
        reference: Optional[str] = None
    ) -> str:
        """
        Create the prompt for the judge LLM.
        
        Args:
            question: The original question
            response: The LLM's response to evaluate
            reference: Optional reference answer for comparison
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert evaluator of LLM responses. Evaluate the following response to a question.

Question: {question}

Response to evaluate: {response}
"""
        
        if reference:
            prompt += f"""
Reference answer (for comparison): {reference}
"""
        
        prompt += """
Evaluate the response on the following criteria (0.0 to 1.0 scale):

1. **Accuracy** (0.0-1.0): Is the information factually correct? Are there any errors or hallucinations?
2. **Relevance** (0.0-1.0): Does the response directly address the question? Is it on-topic?
3. **Completeness** (0.0-1.0): Does the response fully answer the question? Is important information missing?
4. **Clarity** (0.0-1.0): Is the response clear and well-structured? Is it easy to understand?

Provide your evaluation in the following JSON format:
{
    "accuracy": <score>,
    "relevance": <score>,
    "completeness": <score>,
    "clarity": <score>,
    "overall": <average of all scores>,
    "reasoning": "<brief explanation of your scores>"
}

Respond with ONLY the JSON object, no additional text.
"""
        return prompt
    
    def evaluate(
        self,
        question: str,
        response: str,
        reference: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a response using the LLM judge.
        
        Args:
            question: The original question
            response: The LLM's response to evaluate
            reference: Optional reference answer
            
        Returns:
            Dictionary with evaluation scores
        """
        prompt = self._create_judge_prompt(question, response, reference)
        
        try:
            if self.provider == "openai":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise evaluator. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Deterministic evaluation
                    response_format={"type": "json_object"}
                )
                result_text = completion.choices[0].message.content
            else:  # anthropic
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0.0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = message.content[0].text
            
            # Parse JSON response
            import json
            evaluation = json.loads(result_text)
            
            return {
                'accuracy': float(evaluation.get('accuracy', 0.0)),
                'relevance': float(evaluation.get('relevance', 0.0)),
                'completeness': float(evaluation.get('completeness', 0.0)),
                'clarity': float(evaluation.get('clarity', 0.0)),
                'overall': float(evaluation.get('overall', 0.0)),
                'reasoning': evaluation.get('reasoning', '')
            }
        except Exception as e:
            print(f"Error in LLM judge evaluation: {e}")
            # Return default scores on error
            return {
                'accuracy': 0.0,
                'relevance': 0.0,
                'completeness': 0.0,
                'clarity': 0.0,
                'overall': 0.0,
                'reasoning': f'Error: {str(e)}'
            }


def llm_judge_score(
    question: str,
    response: str,
    reference: Optional[str] = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini"
) -> float:
    """
    Convenience function to get overall LLM judge score.
    
    Args:
        question: The original question
        response: The LLM's response to evaluate
        reference: Optional reference answer
        provider: LLM provider to use
        model: Model name
        
    Returns:
        Overall score (0.0 to 1.0)
    """
    judge = LLMJudge(provider=provider, model=model)
    evaluation = judge.evaluate(question, response, reference)
    return evaluation['overall']

