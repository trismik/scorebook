# Adaptive Dataset Splits

## Overview

Scorebook now supports specifying dataset splits for adaptive test datasets. This feature leverages trismik 1.0.2's `split` parameter to allow users to evaluate models on specific dataset splits (e.g., "validation", "test", "train").

## Motivation

Many adaptive test datasets contain multiple splits to separate data used for different purposes:
- **validation**: For hyperparameter tuning and model selection
- **test**: For final model evaluation
- **train**: For understanding baseline performance (less common for evaluation)

Prior to this feature, scorebook had no way to specify which split to use, potentially leading to:
- Using the wrong split for the evaluation purpose
- Confusion about which data was actually evaluated
- Inability to use datasets that don't have a default split configured

## Usage

### Specifying a Split

You can specify a split when defining an adaptive dataset by appending `:split_name` to the dataset string:

```python
from scorebook import evaluate_async

# Specify the validation split explicitly
results = await evaluate_async(
    inference=my_inference_function,
    datasets="my_dataset:adaptive:validation",
    experiment_id="exp_123",
    project_id="proj_456"
)

# Specify the test split explicitly
results = await evaluate_async(
    inference=my_inference_function,
    datasets="my_dataset:adaptive:test",
    experiment_id="exp_123",
    project_id="proj_456"
)
```

### Automatic Split Fallback

If no split is specified, scorebook will automatically select one using the following priority order:

1. **validation** (preferred for most evaluation scenarios)
2. **test** (fallback if validation doesn't exist)
3. **Error** (if neither validation nor test exists)

Example with automatic fallback:

```python
# No split specified - scorebook will try validation, then test
results = await evaluate_async(
    inference=my_inference_function,
    datasets="my_dataset:adaptive",  # No split specified
    experiment_id="exp_123",
    project_id="proj_456"
)
```

The fallback logic will log which split it selected:
```
INFO: Using 'validation' split for my_dataset:adaptive
```

### Multiple Datasets with Different Splits

You can evaluate on multiple splits simultaneously:

```python
results = await evaluate_async(
    inference=my_inference_function,
    datasets=[
        "dataset_a:adaptive:validation",
        "dataset_a:adaptive:test",
        "dataset_b:adaptive",  # Will use fallback
    ],
    experiment_id="exp_123",
    project_id="proj_456"
)
```

## Dataset String Format

The format for specifying adaptive datasets with splits is:

```
<test_id>:adaptive[:<split_name>]
```

Where:
- `<test_id>`: The identifier for the adaptive test dataset
- `:adaptive`: Required marker indicating this is an adaptive dataset
- `:<split_name>`: Optional split name (e.g., "validation", "test", "train")

### Valid Examples

- `"mmlu:adaptive"` - Uses fallback (validation → test)
- `"mmlu:adaptive:validation"` - Uses validation split
- `"mmlu:adaptive:test"` - Uses test split
- `"custom_dataset:adaptive:custom_split"` - Uses custom_split

### Invalid Examples

- `"mmlu:validation:adaptive"` - Wrong order
- `"mmlu:adaptive:validation:extra"` - Too many components
- `"mmlu"` - Missing :adaptive marker (treated as classic dataset)

## Error Handling

### Split Not Found

If you specify a split that doesn't exist, scorebook will raise a `ScoreBookError` with available splits:

```python
# If "train" split doesn't exist for this dataset
results = await evaluate_async(
    inference=my_inference_function,
    datasets="my_dataset:adaptive:train",  # Error!
    experiment_id="exp_123",
    project_id="proj_456"
)
```

Error message:
```
ScoreBookError: Specified split 'train' not found for dataset 'my_dataset:adaptive'.
Available splits: ['validation', 'test']
```

### No Valid Splits Available

If a dataset has no validation or test splits, and no split was specified:

```python
results = await evaluate_async(
    inference=my_inference_function,
    datasets="broken_dataset:adaptive",  # Only has 'train' split
    experiment_id="exp_123",
    project_id="proj_456"
)
```

Error message:
```
ScoreBookError: No suitable split found for dataset 'broken_dataset:adaptive'.
Expected 'validation' or 'test' split. Available splits: ['train']
```

## Implementation Details

### Type Definitions

The implementation adds optional `split` fields to the relevant dataclasses:

```python
@dataclass
class AdaptiveEvalDataset:
    """Represents a dataset configured for adaptive evaluation."""
    name: str
    split: Optional[str] = None


@dataclass
class AdaptiveEvalRunSpec:
    """Specification for an adaptive evaluation run."""
    dataset: str
    dataset_index: int
    hyperparameter_config: Dict[str, Any]
    hyperparameters_index: int
    experiment_id: str
    project_id: str
    split: Optional[str] = None  # Added
    metadata: Optional[Dict[str, Any]] = None
```

### Split Resolution Logic

The `resolve_split_async()` and `resolve_split_sync()` functions in `evaluate_helpers.py` implement the fallback logic:

1. Query the dataset info from trismik to get available splits
2. If user specified a split, validate it exists and use it
3. If no split specified:
   - Try "validation" first
   - Then try "test"
   - Raise error if neither exists

### Trismik API Integration

The resolved split is passed to the trismik client's `run()` method:

```python
trismik_results = await trismik_client.run(
    test_id=adaptive_run_spec.dataset,  # Includes ":adaptive" suffix
    split=resolved_split,  # Resolved split name
    project_id=project_id,
    experiment=experiment_id,
    run_metadata=...,
    item_processor=...,
    return_dict=False,
)
```

## Design Decisions

### Why Validation Before Test?

The fallback prioritizes "validation" over "test" because:

1. **Common Practice**: Validation sets are typically used for model selection and hyperparameter tuning
2. **Safety**: Test sets should be reserved for final evaluation to avoid overfitting
3. **Flexibility**: Users can still explicitly specify "test" if needed

### Why Keep `:adaptive` in test_id?

The `:adaptive` suffix is preserved when calling the trismik API because:

1. The trismik backend expects this format for adaptive datasets
2. It distinguishes adaptive tests from classic tests in the API layer
3. The `split` parameter is separate and independent

### Why Raise Error Instead of Default?

When no validation or test split exists, we raise an error rather than picking an arbitrary split because:

1. **Explicit is better than implicit**: Silently using the wrong split could lead to incorrect conclusions
2. **User awareness**: Forces users to understand their dataset structure
3. **Prevents accidents**: Avoids using training data for evaluation by mistake

## Backward Compatibility

This feature is fully backward compatible:

- Existing code using `"dataset:adaptive"` continues to work with automatic fallback
- The fallback behavior matches expected evaluation best practices
- No breaking changes to existing APIs

## Examples

### Example 1: Hyperparameter Tuning

```python
# Use validation split for hyperparameter tuning
validation_results = await evaluate_async(
    inference=my_model,
    datasets="mmlu:adaptive:validation",
    hyperparameters=[
        {"temperature": 0.7},
        {"temperature": 0.9},
        {"temperature": 1.0},
    ],
    experiment_id="hp_tuning",
    project_id="my_project"
)

# Select best hyperparameters based on validation results
best_temp = select_best(validation_results)

# Final evaluation on test split
test_results = await evaluate_async(
    inference=my_model,
    datasets="mmlu:adaptive:test",
    hyperparameters={"temperature": best_temp},
    experiment_id="final_eval",
    project_id="my_project"
)
```

### Example 2: Cross-Split Analysis

```python
# Compare performance across different splits
results = await evaluate_async(
    inference=my_model,
    datasets=[
        "math_qa:adaptive:validation",
        "math_qa:adaptive:test",
    ],
    experiment_id="split_comparison",
    project_id="my_project"
)

# Analyze if validation performance generalizes to test
for result in results["aggregate_results"]:
    print(f"{result['dataset']}: {result['theta']:.3f} ± {result['std_error']:.3f}")
```

### Example 3: Using Automatic Fallback

```python
# Let scorebook choose the appropriate split
results = await evaluate_async(
    inference=my_model,
    datasets=[
        "dataset_a:adaptive",  # Will use validation
        "dataset_b:adaptive",  # Will use validation
        "dataset_c:adaptive",  # Will use test if no validation exists
    ],
    experiment_id="auto_split",
    project_id="my_project"
)
```

## Troubleshooting

### How do I check available splits?

You can query dataset info using the trismik client:

```python
from scorebook.evaluate.evaluate_helpers import create_trismik_async_client

async with create_trismik_async_client() as client:
    dataset_info = await client.get_dataset_info("my_dataset:adaptive")
    print(f"Available splits: {dataset_info.splits}")
```

### What if I want to use a custom split?

Simply specify it in the dataset string:

```python
results = await evaluate_async(
    inference=my_model,
    datasets="my_dataset:adaptive:custom_split_name",
    experiment_id="exp",
    project_id="proj"
)
```

### Can I disable the fallback behavior?

No, but you can explicitly specify the split to avoid any ambiguity:

```python
# Explicit is better than implicit
results = await evaluate_async(
    inference=my_model,
    datasets="my_dataset:adaptive:validation",  # Explicit split
    experiment_id="exp",
    project_id="proj"
)
```

## Future Enhancements

Potential improvements for future versions:

1. **Split validation**: Warn users if they use test split during hyperparameter tuning
2. **Split metadata**: Include split information in result metadata
3. **Bulk operations**: Helper functions to evaluate on all splits at once
4. **Custom fallback order**: Allow users to configure fallback priority

## Related Documentation

- [Trismik Python Client Documentation](https://docs.trismik.com/python-client)
- [Adaptive Testing Overview](https://docs.trismik.com/adaptive-testing)
- [Scorebook Evaluation Guide](https://docs.scorebook.ai/evaluation)
