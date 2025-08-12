# Scorebook

**A Python library for LLM evaluation**

<p align="center">
  <img alt="Dynamic TOML Badge" src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrismik%2Fscorebook%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=tool.poetry.version&style=flat&label=version">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

Scorebook is a flexible and extensible framework for evaluating Large Language Models (LLMs). It provides clear contracts for data loading, model inference, and metrics computation, making it easy to run comprehensive evaluations across different datasets, models, and metrics.

## ✨ Key Features

- **🔌 Flexible Data Loading**: Support for Hugging Face datasets, CSV, JSON, and Python lists
- **🚀 Model Agnostic**: Works with any model or inference provider
- **📊 Extensible Metric Engine**: Use the metrics we provide or implement your own
- **🔄 Automated Sweeping**: Test multiple model configurations automatically
- **📈 Rich Results**: Export results to JSON, CSV, or structured formats like pandas DataFrames

## 🚀 Quick Start

### Installation

```bash
pip install scorebook
```

For OpenAI integration:
```bash
pip install scorebook[openai]
```

For local model examples:
```bash
pip install scorebook[examples]
```

### Basic Usage

```python
from scorebook import EvalDataset, evaluate
from scorebook.metrics import Accuracy

# 1. Create an evaluation dataset
data = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"}
]

dataset = EvalDataset.from_list(
    name="basic_qa",
    label="answer",
    metrics=[Accuracy],
    data=data
)

# 2. Define your inference function
def my_inference_function(items, **hyperparameters):
    # Your model logic here
    predictions = []
    for item in items:
        # Process each item and generate prediction
        prediction = your_model.predict(item["question"])
        predictions.append(prediction)
    return predictions

# 3. Run evaluation
results = evaluate(my_inference_function, dataset)
print(results)
```

## 📊 Core Components

### 1. Evaluation Datasets

Scorebook supports multiple data sources through the `EvalDataset` class:

#### From Hugging Face
```python
dataset = EvalDataset.from_huggingface(
    "TIGER-Lab/MMLU-Pro",
    label="answer",
    metrics=[Accuracy],
    split="validation"
)
```

#### From CSV
```python
dataset = EvalDataset.from_csv(
    "dataset.csv",
    label="answer",
    metrics=[Accuracy]
)
```

#### From JSON
```python
dataset = EvalDataset.from_json(
    "dataset.json",
    label="answer",
    metrics=[Accuracy]
)
```

#### From Python List
```python
dataset = EvalDataset.from_list(
    name="custom_dataset",
    label="answer",
    metrics=[Accuracy],
    data=[{"question": "...", "answer": "..."}]
)
```

### 2. Model Integration

Scorebook offers two approaches for model integration:

#### Inference Functions
A single function that handles the complete pipeline:

```python
def inference_function(eval_items, **hyperparameters):
    results = []
    for item in eval_items:
        # 1. Preprocessing
        prompt = format_prompt(item)

        # 2. Inference
        output = model.generate(prompt)

        # 3. Postprocessing
        prediction = extract_answer(output)
        results.append(prediction)

    return results
```

#### Inference Pipelines
Modular approach with separate stages:

```python
from scorebook.types.inference_pipeline import InferencePipeline

def preprocessor(item):
    return {"messages": [{"role": "user", "content": item["question"]}]}

def inference_function(processed_items, **hyperparameters):
    return [model.generate(item) for item in processed_items]

def postprocessor(output):
    return output.strip()

pipeline = InferencePipeline(
    model="my-model",
    preprocessor=preprocessor,
    inference_function=inference_function,
    postprocessor=postprocessor
)

results = evaluate(pipeline, dataset)
```

### 3. Metrics System

#### Built-in Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: Accuracy of positive predictions

```python
from scorebook.metrics import Accuracy, Precision

dataset = EvalDataset.from_list(
    name="test",
    label="answer",
    metrics=[Accuracy, Precision],  # Multiple metrics
    data=data
)
```

#### Custom Metrics
Create custom metrics by extending `MetricBase`:

```python
from scorebook.metrics import MetricBase, MetricRegistry

@MetricRegistry.register()
class F1Score(MetricBase):
    @staticmethod
    def score(outputs, labels):
        # Calculate F1 score
        item_scores = [calculate_f1_item(o, l) for o, l in zip(outputs, labels)]
        aggregate_score = {"f1": sum(item_scores) / len(item_scores)}
        return aggregate_score, item_scores

# Use by string name or class
dataset = EvalDataset.from_list(..., metrics=["f1score"])
# or
dataset = EvalDataset.from_list(..., metrics=[F1Score])
```

### 4. Hyperparameter Sweeping

Test multiple configurations automatically:

```python
hyperparameters = {
    "temperature": [0.7, 0.9, 1.0],
    "max_tokens": [50, 100, 150],
    "top_p": [0.8, 0.9]
}

results = evaluate(
    inference_function,
    dataset,
    hyperparameters=hyperparameters,
    score_type="all"
)

# Results include all combinations: 3 × 3 × 2 = 18 configurations
```

### 5. Results and Export

Control result format with `score_type`:

```python
# Only aggregate scores (default)
results = evaluate(model, dataset, score_type="aggregate")

# Only per-item scores
results = evaluate(model, dataset, score_type="item")

# Both aggregate and per-item
results = evaluate(model, dataset, score_type="all")
```

Export results:

```python
# Get EvalResult objects for advanced usage
results = evaluate(model, dataset, return_type="object")

# Export to files
for result in results:
    result.to_json("results.json")
    result.to_csv("results.csv")
```

## 🔧 OpenAI Integration

Scorebook includes built-in OpenAI support for both single requests and batch processing:

```python
from scorebook.inference.openai import responses, batch
from scorebook.types.inference_pipeline import InferencePipeline

# For single requests
pipeline = InferencePipeline(
    model="gpt-4o-mini",
    preprocessor=format_for_openai,
    inference_function=responses,
    postprocessor=extract_response
)

# For batch processing (more efficient for large datasets)
batch_pipeline = InferencePipeline(
    model="gpt-4o-mini",
    preprocessor=format_for_openai,
    inference_function=batch,
    postprocessor=extract_response
)
```

## 📋 Examples

The `examples/` directory contains comprehensive examples:

- **`basic_example.py`**: Local model evaluation with Hugging Face
- **`openai_responses_api.py`**: OpenAI API integration
- **`openai_batch_api.py`**: OpenAI Batch API for large-scale evaluation
- **`hyperparam_sweep.py`**: Hyperparameter optimization
- **`scorebook_showcase.ipynb`**: Interactive Jupyter notebook tutorial

Run an example:

```bash
cd examples/
python basic_example.py --output-dir ./my_results
```

## 🏗️ Architecture

Scorebook follows a modular architecture:

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   EvalDataset   │    │  Inference   │    │     Metrics     │
│                 │    │   Pipeline   │    │                 │
│ • Data Loading  │    │              │    │ • Accuracy      │
│ • HF Integration│    │ • Preprocess │    │ • Precision     │
│ • CSV/JSON      │    │ • Inference  │    │ • Custom        │
│ • Validation    │    │ • Postprocess│    │ • Registry      │
└─────────────────┘    └──────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                   ┌─────────────────────┐
                   │     evaluate()      │
                   │                     │
                   │ • Orchestration     │
                   │ • Progress Tracking │
                   │ • Result Formatting │
                   │ • Export Options    │
                   └─────────────────────┘
```

## 🎯 Use Cases

Scorebook is designed for:

- **Model Benchmarking**: Compare different models on standard datasets
- **Hyperparameter Optimization**: Find optimal model configurations
- **Dataset Analysis**: Understand model performance across different data types
- **A/B Testing**: Compare model versions or approaches
- **Research Experiments**: Reproducible evaluation workflows
- **Production Monitoring**: Track model performance over time

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 About

Scorebook is developed by [Trismik](https://trismik.com) to speed up your LLM evaluation.

---

*For more examples and detailed documentation, check out the Jupyter notebook in `examples/scorebook_showcase.ipynb`*
