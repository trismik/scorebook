# Scorebook Tutorials

This directory contains tutorials, examples, and quickstarts for learning Scorebook - a Python framework for evaluating large language models.

## Directory Structure

```
tutorials/
├── quickstarts/           # Quick start notebooks for getting up and running
├── notebooks/             # Interactive Jupyter notebooks
├── examples/              # Python script examples organized by topic
│   ├── 1-score/           # Scoring pre-computed outputs
│   ├── 2-evaluate/        # Running evaluations
│   ├── 3-evaluation_datasets/  # Loading datasets
│   ├── 4-adaptive_evaluations/ # Trismik adaptive testing
│   ├── 5-upload_results/  # Uploading to Trismik
│   └── 6-providers/       # Cloud provider integrations
└── utils/                 # Helper utilities for examples
```

## Getting Started

### Prerequisites

- Python 3.9+
- Install Scorebook: `pip install scorebook`
- For local model examples: `pip install scorebook[examples]` (includes transformers, torch)
- For cloud examples: `pip install scorebook[openai]` and set `OPENAI_API_KEY`
- For Trismik features: Set `TRISMIK_API_KEY` environment variable

### Quickstarts

Start here for a rapid introduction:

```bash
jupyter notebook tutorials/quickstarts/getting_started.ipynb
```

Available quickstarts:
- `getting_started.ipynb` - Introduction to Scorebook basics
- `classical_evaluations/` - Standard evaluation workflows
- `adaptive_evaluations/` - Trismik's adaptive testing feature

### Notebooks

Interactive tutorials covering core concepts:

```bash
jupyter notebook tutorials/notebooks/
```

| Notebook | Description |
|----------|-------------|
| `1-scoring.ipynb` | Score pre-computed model outputs |
| `2-evaluating.ipynb` | Run full evaluation pipelines |
| `3.1-adaptive_evaluation_phi.ipynb` | Adaptive evaluation with local models |
| `3.2-adaptive_evaluation_gpt.ipynb` | Adaptive evaluation with OpenAI |
| `4-uploading_results.ipynb` | Upload results to Trismik dashboard |

## Examples

Python scripts demonstrating specific features. Run examples from the project root:

```bash
python tutorials/examples/1-score/1-scoring_model_accuracy.py
```
### 1-score: Scoring Pre-computed Outputs

Score model predictions that have already been generated.

| Example | Description |
|---------|-------------|
| `1-scoring_model_accuracy.py` | Score outputs using accuracy metric |
| `2-scoring_model_bleu.py` | Score using BLEU metric |
| `3-scoring_model_f1.py` | Score using F1 metric |
| `4-scoring_model_rouge.py` | Score using ROUGE metric |

### 2-evaluate: Running Evaluations

End-to-end evaluation workflows with inference.

| Example | Description | Requirements |
|---------|-------------|--------------|
| `1-evaluating_local_models.py` | Basic evaluation with local HuggingFace model | - |
| `2-evaluating_local_models_with_batching.py` | Batch processing for improved throughput | - |
| `3-evaluating_cloud_models.py` | Evaluate using OpenAI API | OpenAI API key |
| `4-evaluating_cloud_models_with_batching.py` | OpenAI Batch API for cost savings | OpenAI API key |
| `5-hyperparameter_sweeps.py` | Test multiple hyperparameter configurations | - |
| `6-inference_pipelines.py` | Modular preprocessing/inference/postprocessing | - |

### 3-evaluation_datasets: Loading Datasets

Different ways to load evaluation data.

| Example | Description | Requirements |
|---------|-------------|--------------|
| `1-evaluation_datasets_from_files.py` | Load from JSON/CSV files | - |
| `2-evaluation_datasets_from_huggingface.py` | Load from HuggingFace Hub | OpenAI API key |
| `3-evaluation_datasets_from_huggingface_with_yaml_configs.py` | Use YAML configs for HuggingFace datasets | OpenAI API key |

### 4-adaptive_evaluations: Trismik Adaptive Testing

Efficient evaluation using Item Response Theory (IRT).

| Example | Description | Requirements |
|---------|-------------|--------------|
| `1-adaptive_evaluation.py` | Basic adaptive evaluation | Trismik + OpenAI |
| `2-adaptive_dataset_splits.py` | Adaptive evaluation with dataset splits | Trismik + OpenAI |

### 5-upload_results: Uploading to Trismik

Persist and share results on the Trismik dashboard.

| Example | Description | Requirements |
|---------|-------------|--------------|
| `1-uploading_score_results.py` | Upload `score()` results | Trismik API key |
| `2-uploading_evaluate_results.py` | Upload `evaluate()` results | Trismik API key |
| `3-uploading_your_results.py` | Upload custom results | Trismik API key |

### 6-providers: Cloud Provider Integrations

Batch processing with different cloud providers.

#### AWS Bedrock (`6-providers/aws/`)
- `batch_example.py` - Batch inference with Claude models via AWS Bedrock

**Requirements:** AWS CLI configured, S3 bucket, IAM role for Bedrock

#### Google Cloud Vertex AI (`6-providers/vertex/`)
- `batch_example.py` - Batch inference with Gemini models
- `messages_example.py` - Real-time inference with Gemini

**Requirements:** Google Cloud SDK, Vertex AI enabled project

#### Portkey (`6-providers/portkey/`)
- `batch_example.py` - Batch inference via Portkey gateway
- `messages_example.py` - Real-time inference via Portkey

**Requirements:** Portkey API key, linked provider account

## Additional Resources

- [Scorebook Documentation](https://docs.trismik.com/)
- [Trismik Platform](https://trismik.com)
- [API Reference](https://docs.trismik.com/category/reference/)
- [GitHub Issues](https://github.com/trismik/scorebook/issues)
- Contact support at support@trismik.com