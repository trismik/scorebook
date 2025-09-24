"""Upload classic evaluation run results to Trismik platform."""

import logging
import os
from typing import Any, Dict, List, Optional

from trismik.adaptive_test import AdaptiveTest
from trismik.client_async import TrismikAsyncClient
from trismik.types import (
    TrismikClassicEvalItem,
    TrismikClassicEvalMetric,
    TrismikClassicEvalRequest,
    TrismikClassicEvalResponse,
)

from scorebook.trismik_services.login import get_token
from scorebook.types import ClassicEvalRunResult

logger = logging.getLogger(__name__)


async def upload_classic_eval_run(
    run: ClassicEvalRunResult,
    experiment_id: str,
    project_id: str,
    model: str,
    metadata: Optional[Dict[str, Any]],
) -> TrismikClassicEvalResponse:
    """Upload a classic evaluation run result to Trismik platform.

    Args:
        run: The evaluation run result to upload
        experiment_id: Trismik experiment identifier
        project_id: Trismik project identifier
        model: Model name used for evaluation
        metadata: Optional metadata dictionary

    Returns:
        Response from Trismik API containing the upload result
    """
    service_url = os.environ.get("TRISMIK_SERVICE_URL", "https://api.trismik.com/adaptive-testing")
    runner = AdaptiveTest(
        lambda x: None,
        client=TrismikAsyncClient(service_url=service_url, api_key=get_token()),
    )

    # Create eval items from run_spec items, outputs, and labels
    items: List[TrismikClassicEvalItem] = []
    for idx, (item, output) in enumerate(zip(run.run_spec.items, run.outputs)):
        label = run.run_spec.labels[idx] if idx < len(run.run_spec.labels) else ""

        # Calculate item-level metrics for this item
        item_metrics: Dict[str, Any] = {}
        for metric_name, metric_data in run.scores.items():
            if isinstance(metric_data, dict) and "item_scores" in metric_data:
                if idx < len(metric_data["item_scores"]):
                    item_metrics[metric_name] = metric_data["item_scores"][idx]
            else:
                # If scores is just a single value, use it for all items
                item_metrics[metric_name] = metric_data

        eval_item = TrismikClassicEvalItem(
            datasetItemId=str(idx),
            modelInput=str(item),
            modelOutput=str(output),
            goldOutput=str(label),
            metrics=item_metrics,
        )
        items.append(eval_item)

    # Create eval metrics from run aggregate scores
    metrics: List[TrismikClassicEvalMetric] = []
    for metric_name, metric_data in run.scores.items():
        if isinstance(metric_data, dict) and "aggregate_scores" in metric_data:
            # Handle structured metric data with aggregate scores
            for agg_name, agg_value in metric_data["aggregate_scores"].items():
                metric_id = f"{metric_name}_{agg_name}" if agg_name != metric_name else metric_name
                metric = TrismikClassicEvalMetric(metricId=metric_id, value=agg_value)
                metrics.append(metric)
        else:
            # Handle simple metric data (single value)
            metric = TrismikClassicEvalMetric(metricId=metric_name, value=metric_data)
            metrics.append(metric)

    classic_eval_request = TrismikClassicEvalRequest(
        project_id,
        experiment_id,
        run.run_spec.dataset.name,
        model,
        run.run_spec.hyperparameter_config,
        items,
        metrics,
    )

    response: TrismikClassicEvalResponse = await runner.submit_classic_eval_async(
        classic_eval_request
    )

    run_id: str = response.id
    logger.info(f"Classic eval run uploaded successfully with run_id: {run_id}")

    return response
