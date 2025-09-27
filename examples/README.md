# Scorebook Examples

This directory contains comprehensive examples demonstrating how to use Scorebook for large language model evaluation. The examples are designed to be followed in order, with each building upon concepts from previous examples.

## Overview

Scorebook is a Python framework for evaluating large language models across various datasets and metrics. It supports both local and cloud-based inference, batch processing, hyperparameter sweeps, and integration with the Trismik evaluation platform.

## Getting Started

### Prerequisites

- Python 3.9+
- Install Scorebook: `pip install scorebook`
- For local examples: `pip install scorebook[examples]` (includes transformers, torch, etc.)
- For cloud examples: `pip install scorebook[openai]` and set `OPENAI_API_KEY`
- For Trismik examples: Set `TRISMIK_API_KEY` environment variable

### Basic Workflow

Most Scorebook evaluations follow this pattern:
1. Create an `EvalDataset` from your evaluation data
2. Define an inference function or `InferencePipeline`
3. Run `evaluate()` with your model, dataset, and hyperparameters
4. Analyze results and optionally upload to Trismik dashboard

## Examples

### 1. Simple Evaluation: `example_1_simple_evaluation.py`
**Concepts:** Basic evaluation workflow, EvalDataset creation, inference function

Learn the fundamentals of Scorebook by evaluating a Phi-4 model on basic questions. Shows how to create a dataset from a list, define a simple inference function, and run evaluation with accuracy metrics.

**Key Features:**
- Creating datasets with `EvalDataset.from_list()`
- Simple inference function with transformers pipeline
- Basic hyperparameter configuration

### 2. Evaluation Datasets: `example_2_evaluation_datasets.py`
**Concepts:** Multiple data formats, HuggingFace integration, dataset sampling

Explore different ways to load evaluation datasets including local files (JSON, CSV) and HuggingFace datasets (MMLU, MMLU-Pro). Demonstrates dataset sampling for quick testing on large datasets.

**Key Features:**
- `EvalDataset.from_json()`, `from_csv()`, `from_huggingface()`
- Loading datasets like MMLU and MMLU-Pro
- Using `sample_size` for quick testing
- Evaluating multiple datasets simultaneously

### 3. Inference Pipelines: `example_3_inference_pipelines.py`
**Concepts:** Pipeline architecture, preprocessing, postprocessing

Learn Scorebook's modular pipeline architecture that separates evaluation into three stages: preprocessing (data → model input), inference (model prediction), and postprocessing (model output → final answer).

**Key Features:**
- `InferencePipeline` with separate preprocessor, inference, and postprocessor functions
- Reusable and composable evaluation components
- Structured approach to model evaluation

### 4. Batch Inference: `example_4_batch_inference.py`
**Concepts:** Local batch processing, performance optimization

Optimize evaluation performance by processing multiple items simultaneously. Shows how to implement batch inference for improved GPU utilization and faster evaluation times.

**Key Features:**
- Batch processing for improved throughput
- Performance debugging and monitoring
- Memory-efficient evaluation of large datasets

### 5. Cloud Inference: `example_5_cloud_inference.py`
**Concepts:** OpenAI API integration, cloud-hosted models

Evaluate cloud-hosted models using OpenAI's API. Demonstrates how to adapt inference pipelines for cloud providers with proper error handling and response parsing.

**Key Features:**
- OpenAI API integration with `scorebook.inference.openai.responses`
- Cloud provider authentication
- Parallel inference for improved speed
- Error handling for API responses

### 6. Cloud Batch Inference: `example_6_cloud_batch_inference.py`
**Concepts:** OpenAI Batch API, cost-effective large-scale evaluation

Use OpenAI's Batch API for cost-effective evaluation of large datasets. Shows how to format requests for batch processing and handle asynchronous batch results.

**Key Features:**
- OpenAI Batch API integration with `scorebook.inference.openai.batch`
- Cost-effective evaluation for large datasets
- Asynchronous batch processing
- Batch request formatting and result parsing

### 7. Hyperparameter Sweeps: `example_7_hyperparameters.py`
**Concepts:** Grid search, parameter optimization, systematic evaluation

Systematically evaluate models across multiple hyperparameter configurations. Scorebook automatically generates all combinations (Cartesian product) and runs separate evaluations for each configuration.

**Key Features:**
- Automatic grid generation from hyperparameter lists
- Multiple parameter optimization (temperature, top_p, top_k)
- Systematic configuration comparison
- Performance analysis across parameter space

### 8. Uploading Results: `example_8_uploading_results.py`
**Concepts:** Trismik integration, result persistence, experiment tracking

Upload evaluation results to Trismik's dashboard for persistence, sharing, and analysis. Shows how to organize experiments with metadata and project structure.

**Key Features:**
- Trismik authentication with `login()`
- Experiment and project organization
- Result persistence and sharing
- Metadata attachment for experiment context

### 9. Adaptive Evaluation: `example_9_adaptive_eval.py`
**Concepts:** Adaptive testing, intelligent dataset sampling, Trismik platform

Use Trismik's adaptive evaluation to intelligently select evaluation items based on model performance. Reduces evaluation time while maintaining statistical significance.

**Key Features:**
- Adaptive dataset selection with `:adaptive` suffix
- Intelligent sampling based on model performance
- Integration with Trismik's adaptive algorithms
- Efficient evaluation of large benchmark datasets

## Example Datasets

The `example_datasets/` directory contains sample data files:
- `basic_questions.json/csv`: Simple Q&A pairs for demonstration
- Used across multiple examples for consistency

## Helper Functions

The `example_helpers/` directory provides utility functions used across examples:
- Logging setup and output directory management
- Result saving and experiment organization
- OpenAI model configuration and parsing

## Running Examples

Each example can be run independently:

```bash
# Run a specific example
python examples/example_1_simple_evaluation.py

# Examples with dependencies will show helpful error messages if prerequisites are missing
```

Results are automatically saved to `examples/example_results/` with experiment-specific subdirectories.

## Next Steps

After working through these examples:

1. **Adapt to your use case**: Modify inference functions for your specific models and tasks
2. **Create custom datasets**: Use your own evaluation data with `EvalDataset.from_*` methods
3. **Implement custom metrics**: Define domain-specific evaluation metrics beyond accuracy
4. **Scale up**: Use cloud inference and batch processing for large-scale evaluations
5. **Track experiments**: Set up Trismik integration for result persistence and analysis

## Additional Resources

- [Scorebook Documentation](https://docs.trismik.com/)
- [Trismik Platform](https://trismik.com)
- [API Reference](https://docs.trismik.com/category/reference/)

## Support

For questions or issues:
- Check the [GitHub Issues](https://github.com/trismik/scorebook/issues)
- Contact support at support@trismik.com
