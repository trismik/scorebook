# Scorebook

**A Python library for Model evaluation**

<p align="center">
  <img alt="Dynamic TOML Badge" src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftrismik%2Fscorebook%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=tool.poetry.version&style=flat&label=version">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <a href="https://docs.trismik.com/scorebook/introduction-to-scorebook/" target="_blank" rel="noopener">
    <img alt="Documentation" src="https://img.shields.io/badge/docs-Scorebook-blue?style=flat">
  </a>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

Scorebook provides a flexible and extensible framework for evaluating models such as large language models (LLMs). Easily evaluate any model using evaluation datasets from Hugging Face such as MMLU-Pro, HellaSwag, and CommonSenseQA, or with data from any other source. Evaluations calculate scores for any number of specified metrics such as accuracy, precision, and recall, as well as any custom defined metrics, inluding LLM as a judge (LLMaJ).

## Key Features

- **Model Agnostic**: Evaluate any model, running locally or deployed on the cloud.
- **Dataset Agnostic**: Create evaluation datasets from Hugging Face datasets or any other source.
- **Extensible Metric Engine**: Use the metrics we provide or implement your own.
- **Hyperparameter Sweeping**: Test and compare multiple model hyperparameter configurations automatically.
- **Adaptive Evaluations**: Run Trismik's ultra-fast [adaptive evaluations](https://docs.trismik.com/adaptiveTesting/adaptive-testing-introduction/).
- **Trismik Integration**: Automatically upload your evaluations to [Trismik's platform](https://www.trismik.com/) for storing, managing, and visualizing evaluation results.

## Use Cases

Scorebook's evaluations can be used for:

- **Model Benchmarking**: Compare different models on standard datasets.
- **Model Optimization**: Find optimal model configurations.
- **Iterative Experimentation**: Reproducible evaluation workflows.

## Installation

```bash
pip install scorebook
```

## Overview

Scorebook contains two core functions:
- `score`: Accepts generated model outputs and metrics to calculate scores.
- `evaluate`: Accepts an inference function and evaluation datasets with metrics, to run model inference, generate model outputs, and calculate scores. The evaluate function can be used to run both **classical evaluations** and **adaptive evaluations**.

Both functions by default return a python dict, with two items for aggregate scores, and item scores.
Additionally, both functions have an asynchronous counterpart, `score_async` and `evaluate_async` which are awaitable coroutines that allow asynchronous inference functions or metric score functions.

Scorebook also contains integration with Trismik's platform for adaptive evaluations, and the use of Trismik's experimentation dashboard.
The results of `score` and `evaluate` functions can be passed to Scorebook's `upload_result` function, to upload evaluation results to Trismik's dashboard. These results can also be automatically uploaded, when logged in (`scorebook.login("TRISMIK-API-KEY")`) and a project id is provided.

---

### Scoring Models with `score`

Scorebooks score function can be used to evaluate pre-generated model outputs

**Scoring Example**:
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

**Score Results**:

The following shows a snippet of how results from `score` are structured. "item_results" will contain a dict object for each evaluation item.
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
        }...
```

---

### _Classical_ Evaluations with `evaluate`

Running a _classical_ evaluation with scorebook, runs model inference over every item in an evaluation dataset, to generate an output for each evaluation item.
Once the results are collected, they are then scored via the evaluation dataset's specified metrics, to return a result quantifying the model's performance.

Evaluate functions can accept hyperparameters, which are passed to the inference function as kwargs.
A single hyperparameter configuration, a list of configurations, or a grid of configurations to be expanded, can be provided, to evaluate a model across multiple hyperparameter configurations in a single evaluation call.
To view an example of this, see [hyperparameter sweeps]().

**Classical Evaluation example**:
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

**Evaluation Results**:

The following shows a snippet of how results from `evaluate` are structured.
The "item_results" will contain a dict object for each evaluation item.

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
        }...
```

---

### _Adaptive_ Evaluations with `evaluate`

To run an adaptive evaluation, simpley use one of [Trismik's adaptive datasets](). Trismik's computerized adaptive testing (CAT) algorithim, will inteligently select the next evaluation item to provide, to estimate a model's ability score, theta value, with a minimal standard deviation and number of evaluation items used. To run adaptive evaluations, a Trismik API key is required.

**Adaptive Evaluation Example**:
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
    experiment_id = "TRISMIK_PROJECT_ID",    # Optional: An identifier to upload this run under
)
```

**Adaptive Evaluation Results**:

When running an adaptive evaluation, each run will return two interconnected metrics, _theta_ a standard error value. The theta value is an estimation of a model's ability that is highly correlated with the metric used for a given dataset.
Theta values are unbounded, a value of 0 represents a model getting 50% of predictions correct, and higher or lower scores represent better or worse ability respectively.
```json
{
    "aggregate_results": [
        {
            "dataset": "trismik/headQA:adaptive",
            "experiment_id": "TRISMIK_PROJECT_ID",
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
---
## Core Components

### Metrics

Metrics are used to quantify the performance of a model, and Scorebook contains built in metrics, are allows for the creation of custom metrics.
Every metric extends the `MetricBase` class, which is a callable class, which when called, executes the metric's internal `score` function,  which accepts a list of model outputs and labels, and returns a list of scores for each evaluation item, and a dictionary for aggregate scores.

#### Using Metrics

The `score` function and the `EvalDataset` class can except a single, or multiple registered metric when called or instantiated respectively. They can be passed as objects, or as strings, where the associated string name for a metric is it's class name in lowercase.

```python
# Both are valid useages of metrics.

results = scorebook.score(evaluation_items, [Accuracy, Precision])

results = scorebook.score(evaluation_items, ["accuracy", "precision"])
```

#### Built-In Metrics

| Metric        | Sync/Async | Aggregate Scores                | Item Scores                           |
|---------------|------------|---------------------------------|---------------------------------------|
| `Accuracy`    | Sync       | `Boolean`: Exact match between output and label | `Float`: Percentage of correct outputs  |

#### Custom Metrics
Create custom metrics by extending `MetricBase` and defining a `score` function.

The metric's score function can use any calculation to generate scores, the only constraints when defining a metric are that its signature must match the following:

**Metric Score Method Arguments**:

- `outputs: List[Any]`
- `labels: List[Any`

**Metric Score Method Returns**:

- `scores: Dict[str: Any]`

The dict scores returned by a metric's score method, should contain two keys, "aggregate_scores" and "item_scores" with a dict of aggrgate scores, and list of item scores respectively.
See how the [accuracy](https://github.com/trismik/scorebook/blob/main/src/scorebook/metrics/accuracy.py) metric is implemented for guidance.

The example below shows the creation and registration of a spell checking metric, which generates a scores for the percentage of correctly spelt words.

```python
from spellchecker import SpellChecker # pip install pyspellchecker
from scorebook.metrics import MetricBase, MetricRegistry

@MetricRegistry.register()
class SpellCheck(MetricBase):
    """
    Spell-check accuracy over text outputs.

    Formula: accuracy = 1 − (misspelled_tokens / total_tokens).
    If an output has no tokens, its score is 1.0.
    """
    _sp = SpellChecker("en")

    @staticmethod
    def score(outputs, labels):

        items = []
        for s in outputs:
            words = [w.strip(".,;:!?\"'()").lower() for w in (s or "").split() if w.strip()]
            acc = 1.0 if not words else 1.0 - len(SpellCheck._sp.unknown(words)) / len(words)
            items.append(acc)

        return {"spellcheck": sum(items)/len(items) if items else 0.0}, items
```

### Evaluation Datasets

In Scorebook, an evaluation dataset, represented by the `EvalDataset` class, is a dataset for model evaluation, which specifies which feature values are to be input into the model, and which feature values are to be considered as gold labels for scoring model output against. Evaluation datasets also list associated metrics to be used for scoring during evaluation.

#### Example Evaluation Dataset:

- Dataset Name: Basic Questions
- Metrics: Accuracy

| Input                             | Label                 |
|-----------------------------------|-----------------------|
| "What is 2 + 2?"                  | "4"                   |
| "What is the capital of France?"  | "Paris"               |
| "Who wrote Romeo and Juliet?"     | "William Shakespeare" |

#### Creating Evaluation Datasets

Evaluation Datasets can be created using the `EvalDataset`'s factory methods:

- `EvalDataset.from_list`: provided a list of evaluation items as Python dicts.
- `EvalDataset.from_json`: provided the path to a valid json file, containing structured evaluation items.
- `EvalDataset.from_csv`: provided the path to a valid csv file, containing structured evaluation items.

```python
"""Creating an evaluation dataset from a list of evaluation items"""

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
```

#### Using Hugging Face Datasets

Evaluation datasets can be created from Hugging Face datasets with:

- `EvalDataset.from_huggingface`: For Hugging Face Datasets that already are structured for evaluation.
- `EvalDataset.from_yaml`: For the use of .yaml config files that specify how to load and prepare Hugging Face datasets.

### Inference Functions
The `evaluate` functions require an inference function that encapsulates a model, to generate outputs for evaluation items.
There is flexibility in how an inference function is implemented, it can encapsulate local or cloud-based models, and by synchronous or asynchronous.
Typically an inference function will prepare input values, by wrapping them in a message structure,
passing prepared items to the model for inference,
and extracting the output value from the structured response.
The only requirements are in the functions signature.

**Inference Function Args**:

- `inputs: List[Any]` The list of input values from an evaluation dataset.
- `hyperparameters: Dict[str, Any]` Hyperparameters as a dict, that can be optionally used.

**Inference Function Returns**:

- `outputs: Any` Model outputs, prepared for scoring against label values.


**Example Inference Function**:
```python
def inference_function(inputs, **hyperparameters):
    results = []
    for input in inputs:
        # 1. Preprocessing
        prompt = format_prompt(input)

        # 2. Inference
        output = model.generate(prompt)

        # 3. Postprocessing
        prediction = extract_answer(output)
        results.append(prediction)

    return results
```

### Inference Pipeline

For more complex evaluation workflows, `InferencePipeline` provides a modular approach that separates inference into three distinct stages:

1. **Preprocessing**: Transform dataset items into model-ready input format
2. **Inference**: Execute model predictions on preprocessed data
3. **Postprocessing**: Extract final answers from raw model outputs

This separation of concerns improves code reusability and maintainability. The pipeline automatically adapts to synchronous or asynchronous execution based on the inference function provided, and can be passed directly to `evaluate()` just like standard inference functions.

**Example InferencePipeline**:
```python
from scorebook import InferencePipeline

def preprocessor(item, **hyperparameters):
    # Convert dataset item to model input format
    return {"messages": [
        {"role": "system", "content": hyperparameters["system_message"]},
        {"role": "user", "content": item}
    ]}

def inference_function(processed_items, **hyperparameters):
    # Run model inference on preprocessed items
    return [model.generate(item, temperature=hyperparameters["temperature"])
            for item in processed_items]

def postprocessor(output, **hyperparameters):
    # Extract final answer from model output
    return output.strip()

pipeline = InferencePipeline(
    model="model-name",
    preprocessor=preprocessor,
    inference_function=inference_function,
    postprocessor=postprocessor
)

results = evaluate(pipeline, dataset, hyperparameters={"temperature": 0.7, "system_message": "..."})
```

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

---
