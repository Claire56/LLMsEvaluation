"""
Evaluation pipeline for testing multiple prompts on Q&A pairs.
"""

from .evaluator import Evaluator
from .prompt_templates import PromptTemplates

__all__ = ['Evaluator', 'PromptTemplates']

