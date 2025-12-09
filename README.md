# LLM Evaluation Framework

A comprehensive guide and implementation for evaluating Large Language Model (LLM) outputs.

## 📚 Learning Objectives

This project teaches you about:

1. **Evaluation Frameworks**: How to systematically measure LLM performance
2. **Output Quality Metrics**: Relevance, accuracy, and hallucination detection
3. **LLMs-as-Judges**: Using LLMs to evaluate other LLMs
4. **Text Similarity Metrics**: ROUGE and BLEU scores
5. **Custom Metrics**: Building domain-specific evaluation criteria
6. **Cost & Latency Tracking**: Understanding the trade-offs between different prompts

## 🎯 What You'll Learn

### 1. Relevance
Relevance measures how well the LLM's response addresses the question. A relevant answer:
- Directly answers the question
- Stays on topic
- Provides appropriate detail level

### 2. Accuracy
Accuracy measures factual correctness. An accurate answer:
- Contains correct information
- Doesn't contradict known facts
- Provides verifiable claims

### 3. Hallucinations
Hallucinations are when LLMs generate:
- Factually incorrect information
- Information not present in the training data
- Contradictory statements
- Fabricated details

### 4. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
ROUGE measures overlap between generated and reference text:
- **ROUGE-1**: Unigram overlap (word-level)
- **ROUGE-2**: Bigram overlap (phrase-level)
- **ROUGE-L**: Longest common subsequence (sentence structure)

**Use Case**: Best for summarization, translation, and text generation tasks.

### 5. BLEU (Bilingual Evaluation Understudy)
BLEU measures precision of n-grams:
- Compares generated text to reference text
- Scores range from 0 to 1 (higher is better)
- Penalizes overly short or repetitive outputs

**Use Case**: Best for translation and text generation where precision matters.

### 6. LLMs-as-Judges
Using a judge LLM to evaluate another LLM's output:
- **Advantages**: Understands context, can evaluate nuanced quality
- **Approach**: Provide question, answer, and reference to judge LLM
- **Output**: Numerical score or categorical rating

**Use Case**: When you need semantic understanding beyond n-gram matching.

## 🏗️ Project Structure

```
LLM_Evaluation/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── metrics/                  # Evaluation metrics
│   ├── __init__.py
│   ├── rouge_bleu.py        # ROUGE and BLEU implementations
│   ├── llm_judge.py         # LLM-as-judge evaluator
│   └── custom_metrics.py    # Custom evaluation functions
├── pipeline/                 # Evaluation pipeline
│   ├── __init__.py
│   ├── evaluator.py         # Main evaluation orchestrator
│   └── prompt_templates.py  # Different prompt variations
├── data/                     # Datasets
│   ├── qa_dataset.json      # 100 Q&A pairs
│   └── generate_dataset.py  # Script to generate sample data
├── dashboard/                # Visualization
│   ├── __init__.py
│   └── visualizer.py        # Dashboard generator
└── main.py                   # Main execution script
```

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

2. **Set up API keys** (create `.env` file):
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

3. **Run evaluation**:
```bash
python main.py
```

4. **View dashboard**:
The dashboard will be generated as `evaluation_dashboard.html`

## 📊 Evaluation Pipeline

The pipeline evaluates 5 different prompts on 100 Q&A pairs:

1. **Baseline Prompt**: Simple, direct question
2. **Detailed Prompt**: Asks for comprehensive answer
3. **Few-Shot Prompt**: Includes examples
4. **Chain-of-Thought Prompt**: Asks for reasoning steps
5. **Structured Prompt**: Requests formatted output

For each prompt, we measure:
- **Accuracy**: LLM-as-judge score (0-1)
- **Relevance**: ROUGE-L score
- **Latency**: Time to generate response (seconds)
- **Cost**: Estimated API cost per response

## 📈 Understanding the Metrics

### ROUGE Scores
- **0.0-0.3**: Low overlap, likely irrelevant
- **0.3-0.5**: Moderate overlap, somewhat relevant
- **0.5-0.7**: Good overlap, relevant
- **0.7-1.0**: High overlap, very relevant

### BLEU Scores
- **0.0-0.3**: Low precision, many errors
- **0.3-0.5**: Moderate precision
- **0.5-0.7**: Good precision
- **0.7-1.0**: High precision, very close to reference

### LLM Judge Scores
- **0.0-0.4**: Poor quality (inaccurate, irrelevant, or hallucinated)
- **0.4-0.6**: Acceptable quality
- **0.6-0.8**: Good quality
- **0.8-1.0**: Excellent quality

## 🎓 Key Concepts

### Why Multiple Metrics?
Different metrics capture different aspects:
- **ROUGE**: Word/phrase overlap (surface-level similarity)
- **BLEU**: Precision of n-grams (translation quality)
- **LLM Judge**: Semantic understanding (human-like evaluation)

### When to Use What?
- **ROUGE**: Summarization, abstractive tasks
- **BLEU**: Translation, generation tasks
- **LLM Judge**: When you need nuanced quality assessment
- **Custom Metrics**: Domain-specific requirements

### Cost Considerations
- Longer prompts = more tokens = higher cost
- More detailed responses = higher latency
- Balance quality vs. efficiency

## 🔬 Advanced Topics

### Hallucination Detection
1. **Fact-checking**: Compare against knowledge base
2. **Consistency checks**: Verify internal consistency
3. **Confidence scores**: Ask LLM for confidence level
4. **Multi-model verification**: Compare across models

### Custom Metrics
Build metrics for:
- Domain-specific terminology
- Style consistency
- Safety and bias
- Task-specific success criteria

## 📝 Example Usage

```python
from pipeline.evaluator import Evaluator
from metrics.rouge_bleu import calculate_rouge, calculate_bleu
from metrics.llm_judge import llm_judge_score

# Initialize evaluator
evaluator = Evaluator()

# Evaluate a single Q&A pair
question = "What is machine learning?"
reference = "Machine learning is a subset of AI..."
response = evaluator.get_llm_response(question, prompt_template)

# Calculate metrics
rouge_score = calculate_rouge(response, reference)
bleu_score = calculate_bleu(response, reference)
judge_score = llm_judge_score(question, response, reference)
```

## 🎯 Next Steps

1. Experiment with different prompt templates
2. Add more evaluation metrics
3. Test on your own datasets
4. Implement hallucination detection
5. Build domain-specific evaluators

## 📚 Further Reading

- [ROUGE Paper](https://aclanthology.org/W04-1013/)
- [BLEU Paper](https://aclanthology.org/P02-1040/)
- [LLM-as-Judge Paper](https://arxiv.org/abs/2306.05685)
- [Evaluating LLMs](https://arxiv.org/abs/2303.16634)

