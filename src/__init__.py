"""
Scorebook: A framework for evaluating machine learning model outputs.

This package provides tools for:
- Creating and managing evaluation datasets
- Running model evaluations with custom metrics
- Supporting multiple dataset formats (HuggingFace, JSON, CSV)
- Extensible metric system for model performance assessment

Example:
    from scorebook import EvalDataset, evaluate

    dataset = EvalDataset.from_json("data.json", label="answer")
    results = evaluate(model_fn, dataset, metrics=["precision"])

For more information, visit: github.com/trismik/scorebook

"""
