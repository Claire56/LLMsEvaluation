"""
Different prompt templates for testing prompt engineering strategies.

We'll test 5 different prompt styles:
1. Baseline: Simple, direct
2. Detailed: Asks for comprehensive answer
3. Few-shot: Includes examples
4. Chain-of-thought: Asks for reasoning
5. Structured: Requests formatted output
"""

from typing import Dict


class PromptTemplates:
    """Collection of prompt templates for evaluation."""
    
    @staticmethod
    def baseline(question: str) -> str:
        """
        Baseline prompt: Simple and direct.
        
        This is the control - minimal prompt engineering.
        """
        return f"{question}"
    
    @staticmethod
    def detailed(question: str) -> str:
        """
        Detailed prompt: Asks for comprehensive answer.
        
        Tests if asking for more detail improves quality.
        """
        return f"""Please provide a comprehensive and detailed answer to the following question.

Question: {question}

Provide a thorough explanation that covers all relevant aspects of the topic."""
    
    @staticmethod
    def few_shot(question: str) -> str:
        """
        Few-shot prompt: Includes examples.
        
        Tests if examples improve understanding and output quality.
        """
        return f"""Here are some examples of good Q&A pairs:

Example 1:
Q: What is photosynthesis?
A: Photosynthesis is the process by which plants convert light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen.

Example 2:
Q: What is the capital of France?
A: The capital of France is Paris, a major European city known for its culture, history, and landmarks like the Eiffel Tower.

Now answer this question in a similar style:
Q: {question}
A:"""
    
    @staticmethod
    def chain_of_thought(question: str) -> str:
        """
        Chain-of-thought prompt: Asks for reasoning steps.
        
        Tests if explicit reasoning improves accuracy and reduces hallucinations.
        """
        return f"""Answer the following question by first thinking through your reasoning step by step, then providing your final answer.

Question: {question}

Think step by step:
1. What is this question asking?
2. What information do I need to answer it?
3. What is the correct answer?

Final Answer:"""
    
    @staticmethod
    def structured(question: str) -> str:
        """
        Structured prompt: Requests formatted output.
        
        Tests if structured output improves clarity and completeness.
        """
        return f"""Please answer the following question in a structured format.

Question: {question}

Provide your answer in the following format:

**Main Answer:**
[Your direct answer here]

**Key Points:**
- [Point 1]
- [Point 2]
- [Point 3]

**Additional Context:**
[Any relevant additional information]

Begin your response:"""
    
    @classmethod
    def get_all_templates(cls) -> Dict[str, callable]:
        """
        Get all prompt templates as a dictionary.
        
        Returns:
            Dictionary mapping template names to functions
        """
        return {
            'baseline': cls.baseline,
            'detailed': cls.detailed,
            'few_shot': cls.few_shot,
            'chain_of_thought': cls.chain_of_thought,
            'structured': cls.structured
        }
    
    @classmethod
    def get_template_names(cls) -> list:
        """Get list of all template names."""
        return list(cls.get_all_templates().keys())

