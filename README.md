<h1 align="center">Scorebook</h1>

<p align="center"><strong>A Python library for Model evaluation</strong></p>

<p align="center">
  <img alt="Dynamic TOML Badge" src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrismik%2Fscorebook%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=tool.poetry.version&style=flat&label=version">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <a href="https://docs.trismik.com/scorebook/introduction-to-scorebook/" target="_blank" rel="noopener">
    <img alt="Documentation" src="https://img.shields.io/badge/docs-Scorebook-blue?style=flat">
  </a>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

Scorebook provides a flexible and extensible framework for evaluating models such as large language models (LLMs). Easily evaluate any model using evaluation datasets from Hugging Face such as MMLU-Pro, HellaSwag, and CommonSenseQA, or with data from any other source. Evaluations calculate scores for any number of specified metrics such as accuracy, precision, and recall, as well as any custom defined metrics, including LLM as a judge (LLMaJ).

## Use Cases

Scorebook's evaluations can be used for:

- **Model Benchmarking**: Compare different models on standard datasets.
- **Model Optimization**: Find optimal model configurations.
- **Iterative Experimentation**: Reproducible evaluation workflows.

## Key Features

- **Model Agnostic**: Evaluate any model, running locally or deployed on the cloud.
- **Dataset Agnostic**: Create evaluation datasets from Hugging Face datasets or any other source.
- **Extensible Metric Engine**: Use the Scorebook's built-in or implement your own.
- **Hyperparameter Sweeping**: Evaluate over multiple model hyperparameter configurations.
- **Adaptive Evaluations**: Run Trismik's ultra-fast [adaptive evaluations](https://docs.trismik.com/adaptiveTesting/adaptive-testing-introduction/).
- **Trismik Integration**: Upload evaluations to [Trismik's platform](https://www.trismik.com/).

## Installation

```bash
pip install scorebook
```

## Scoring Models Output

Scorebooks score function can be used to evaluate pre-generated model outputs.

### Score Example
```python
from scorebook import score
from scorebook.metrics import Accuracy

# 1. Prepare a list of generated model outputs and labels
model_predictions = [
    {"input": "What is 2 + 2?", "output": "4", "label": "4"},
    {"input": "What is the capital of France?", "output": "London", "label": "Paris"},
    {"input": "Who wrote Romeo and Juliette?", "output": "William Shakespeare", "label": "William Shakespeare"},
    {"input": "What is the chemical symbol for gold?", "output": "Au", "label": "Au"},
]

# 2. Score the model's predictions against labels using metrics
results = score(
    items = model_predictions,
    metrics = Accuracy,
)
```

### Score Results:
```json
{
    "aggregate_results": [
        {
            "dataset": "scored_items",
            "accuracy": 0.75
        }
    ],
    "item_results": [
        {
            "id": 0,
            "dataset": "scored_items",
            "input": "What is 2 + 2?",
            "output": "4",
            "label": "4",
            "accuracy": true
        }
        // ... additional items
    ]
}
```

## _Classical_ Evaluations

Running a classical evaluation in Scorebook executes model inference on every item in the dataset, then scores the generated outputs using the dataset’s specified metrics to quantify model performance.

### Classical Evaluation example:
```python
from scorebook import evaluate, EvalDataset
from scorebook.metrics import Accuracy

# 1. Create an evaluation dataset
evaluation_items = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"}
]

evaluation_dataset = EvalDataset.from_list(
    name = "basic_questions",
    items = evaluation_items,
    input = "question",
    label = "answer",
    metrics = Accuracy,
)

# 2. Define an inference function - This is a pseudocode example
def inference_function(inputs: List[Any], **hyperparameters):

    # Create or call a model
    model = Model()
    model.temperature = hyperparameters.get("temperature")

    # Call model inference
    model_outputs = model(inputs)

    # Return outputs
    return model_outputs

# 3. Run evaluation
evaluation_results = evaluate(
    inference_function,
    evaluation_dataset,
    hyperparameters = {"temperature": 0.7}
)
```

### Evaluation Results:
```json
{
    "aggregate_results": [
        {
            "dataset": "basic_questions",
            "temperature": 0.7,
            "accuracy": 1.0,
            "run_completed": true
        }
    ],
    "item_results": [
        {
            "id": 0,
            "dataset": "basic_questions",
            "input": "What is 2 + 2?",
            "output": "4",
            "label": "4",
            "temperature": 0.7,
            "accuracy": true
        }
        // ... additional items
    ]
}
```

### _Adaptive_ Evaluations with `evaluate`

To run an adaptive evaluation, use a Trismik adaptive dataset
The CAT algorithm dynamically selects items to estimate the model’s ability (θ) with minimal standard error and fewest questions.

### Adaptive Evaluation Example
```python
from scorebook import evaluate, login

# 1. Log in with your Trismik API key
login("TRISMIK_API_KEY")

# 2. Define an inference function
def inference_function(inputs: List[Any], **hyperparameters):

    # Create or call a model
    model = Model()

    # Call model inference
    outputs = model(inputs)

    # Return outputs
    return outputs

# 3. Run an adaptive evaluation
results = evaluate(
    inference_function,
    datasets = "trismik/headQA:adaptive",    # Adaptive datasets have the ":adaptive" suffix
    project_id = "TRISMIK_PROJECT_ID",       # Required: Create a project on your Trismik dashboard
    experiment_id = "TRISMIK_EXPERIMENT_ID", # Optional: An identifier to upload this run under
)
```

### Adaptive Evaluation Results
```json
{
    "aggregate_results": [
        {
            "dataset": "trismik/headQA:adaptive",
            "experiment_id": "TRISMIK_EXPERIMENT_ID",
            "project_id": "TRISMIK_PROJECT_ID",
            "run_id": "RUN_ID",
            "score": {
                "theta": 1.2,
                "std_error": 0.20
            },
            "responses": null
        }
    ],
    "item_results": []
}
```

## Metrics

| Metric     | Sync/Async | Aggregate Scores                                 | Item Scores                             |
|------------|------------|--------------------------------------------------|-----------------------------------------|
| `Accuracy` | Sync       | `Float`: Percentage of correct outputs           | `Boolean`: Exact match between output and label |
| `ROUGE`    | Sync       | `Dict[str, Float]`: Average F1 scores per ROUGE type | `Dict[str, Float]`: F1 scores per ROUGE type |


## Tutorials

For local more detailed and runnable examples:
```bash
pip install scorebook[examples]
```

The `tutorials/` directory contains comprehensive tutorials as notebooks and code examples:

- **`tutorials/notebooks`**: Interactive Jupyter Notebooks showcasing Scorebook's capabilities.
- **`tutorials/examples`**: Runnable Python examples incrementally implementing Scorebook's features.

**Run a notebook:**
```bash
jupyter notebook tutorials/notebooks
```

**Run an example:**
```bash
python3 tutorials/examples/1-score/1-scoring_model_accuracy.py
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About

Scorebook is developed by [Trismik](https://trismik.com) to simplify and speed up your LLM evaluations.
