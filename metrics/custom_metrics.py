"""
Custom evaluation metrics for domain-specific evaluation.

These metrics go beyond standard text similarity and measure
aspects like length appropriateness, keyword coverage, and
hallucination indicators.
"""

import re
from typing import Dict, List, Set
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))


def calculate_length_ratio(
    generated_text: str,
    reference_text: str
) -> float:
    """
    Calculate the ratio of generated text length to reference length.
    
    Useful for detecting:
    - Overly verbose responses (ratio >> 1.0)
    - Too brief responses (ratio << 1.0)
    
    Args:
        generated_text: Generated text
        reference_text: Reference text
        
    Returns:
        Length ratio (generated / reference)
    """
    gen_length = len(word_tokenize(generated_text))
    ref_length = len(word_tokenize(reference_text))
    
    if ref_length == 0:
        return 1.0 if gen_length == 0 else float('inf')
    
    return gen_length / ref_length


def calculate_keyword_overlap(
    generated_text: str,
    reference_text: str,
    min_word_length: int = 4
) -> Dict[str, float]:
    """
    Calculate keyword overlap between generated and reference text.
    
    Keywords are non-stopwords of minimum length.
    Useful for measuring semantic relevance beyond exact matches.
    
    Args:
        generated_text: Generated text
        reference_text: Reference text
        min_word_length: Minimum length for a word to be considered a keyword
        
    Returns:
        Dictionary with overlap metrics
    """
    def extract_keywords(text: str) -> Set[str]:
        words = word_tokenize(text.lower())
        keywords = {
            word for word in words
            if len(word) >= min_word_length
            and word not in stop_words
            and word.isalnum()
        }
        return keywords
    
    gen_keywords = extract_keywords(generated_text)
    ref_keywords = extract_keywords(reference_text)
    
    if len(ref_keywords) == 0:
        return {
            'overlap_ratio': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    intersection = gen_keywords & ref_keywords
    union = gen_keywords | ref_keywords
    
    overlap_ratio = len(intersection) / len(union) if union else 0.0
    precision = len(intersection) / len(gen_keywords) if gen_keywords else 0.0
    recall = len(intersection) / len(ref_keywords) if ref_keywords else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'overlap_ratio': overlap_ratio,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gen_keywords': len(gen_keywords),
        'ref_keywords': len(ref_keywords),
        'common_keywords': len(intersection)
    }


def detect_hallucination_indicators(
    generated_text: str,
    reference_text: str
) -> Dict[str, any]:
    """
    Detect potential hallucination indicators in generated text.
    
    This is a simple heuristic-based approach. For production,
    you'd want more sophisticated methods (fact-checking, consistency
    checks, confidence scores, etc.).
    
    Args:
        generated_text: Generated text to check
        reference_text: Reference text for comparison
        
    Returns:
        Dictionary with hallucination indicators
    """
    indicators = {
        'has_uncertainty_phrases': False,
        'has_specific_dates': False,
        'has_specific_numbers': False,
        'contradiction_keywords': [],
        'confidence_phrases': []
    }
    
    # Check for uncertainty phrases (might indicate hedging)
    uncertainty_patterns = [
        r'\b(might|may|could|possibly|perhaps|maybe)\b',
        r'\b(uncertain|unclear|unknown|unsure)\b'
    ]
    for pattern in uncertainty_patterns:
        if re.search(pattern, generated_text, re.IGNORECASE):
            indicators['has_uncertainty_phrases'] = True
            break
    
    # Check for overly specific claims (dates, numbers)
    date_pattern = r'\b\d{4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    if re.search(date_pattern, generated_text, re.IGNORECASE):
        indicators['has_specific_dates'] = True
    
    number_pattern = r'\b\d+\.\d+%|\b\d+%\b'
    if re.search(number_pattern, generated_text):
        indicators['has_specific_numbers'] = True
    
    # Check for contradiction keywords
    contradiction_keywords = ['however', 'but', 'although', 'despite', 'contrary']
    found_contradictions = [
        word for word in contradiction_keywords
        if word in generated_text.lower()
    ]
    indicators['contradiction_keywords'] = found_contradictions
    
    # Check for high confidence phrases (might indicate overconfidence)
    confidence_phrases = [
        'definitely', 'certainly', 'absolutely', 'without doubt',
        'proven', 'established fact', 'known to be'
    ]
    found_confidence = [
        phrase for phrase in confidence_phrases
        if phrase in generated_text.lower()
    ]
    indicators['confidence_phrases'] = found_confidence
    
    # Calculate a simple hallucination risk score
    risk_score = 0.0
    if indicators['has_specific_dates'] and not any(date in reference_text for date in re.findall(date_pattern, generated_text)):
        risk_score += 0.3
    if indicators['has_specific_numbers'] and not any(num in reference_text for num in re.findall(number_pattern, generated_text)):
        risk_score += 0.3
    if len(indicators['confidence_phrases']) > 2:
        risk_score += 0.2
    if len(indicators['contradiction_keywords']) > 1:
        risk_score += 0.2
    
    indicators['hallucination_risk'] = min(risk_score, 1.0)
    
    return indicators


def calculate_comprehensive_metrics(
    generated_text: str,
    reference_text: str
) -> Dict[str, any]:
    """
    Calculate all custom metrics in one call.
    
    Args:
        generated_text: Generated text
        reference_text: Reference text
        
    Returns:
        Dictionary with all custom metrics
    """
    return {
        'length_ratio': calculate_length_ratio(generated_text, reference_text),
        'keyword_overlap': calculate_keyword_overlap(generated_text, reference_text),
        'hallucination_indicators': detect_hallucination_indicators(generated_text, reference_text)
    }

