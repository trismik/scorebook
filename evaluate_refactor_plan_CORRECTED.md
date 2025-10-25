# Evaluate Refactor Plan - Full score_async() Integration (CORRECTED)

## Executive Summary

Refactor `evaluate_async.py` to use `score_async()` directly, eliminating **all** duplicated scoring and upload logic. The sync version (`evaluate.py`) will be **auto-generated** using `unasync`.

> **Key Update**: This plan now correctly handles sync generation via `unasync` and includes necessary test updates.

## Critical Insight

**score_async() already does both scoring AND uploading** in a single, well-tested function. Currently, evaluate_async() duplicates this entire flow:

```python
# Current evaluate_async flow (DUPLICATED):
execute_classic_eval_run() -> scores using score_metrics()
worker() -> uploads using upload_classic_run_results()

# Proposed flow (INTEGRATED):
execute_classic_eval_run() -> calls score_async() which does both
```

> **Important**: The sync version (`evaluate.py`) is auto-generated from `evaluate_async.py` using `unasync`. We only modify the async version and run `python scripts/generate_sync.py`.

## Benefits of Full Integration

✅ **Eliminates ~137 lines of code**
✅ **Single source of truth** for scoring + uploading logic
✅ **Automatic bug fixes** - changes to score_async() automatically benefit evaluate()
✅ **Simpler code** - execute_classic_eval_run() delegates to score_async()
✅ **Better testability** - only test scoring/upload logic in one place
✅ **Future-proof** - new score_async() features automatically work in evaluate()
✅ **Auto-generated sync version** - no manual sync/async maintenance

## Current Code Duplication

### Scoring Duplication (~25 lines)
- `score_metrics()` in evaluate_helpers.py (lines 267-292)
- `calculate_metric_scores_async()` in score_helpers.py (lines 63-97)
- **Both iterate metrics and call metric.score()**

### Upload Duplication (~90 lines)
- `upload_classic_run_results()` in evaluate_async.py (lines 316-408)
- `upload_run_result_async()` in upload_results.py (lines 160-290)
- **Both build TrismikClassicEvalRequest and submit to API**

**Total: ~115 lines of duplicated logic that can be eliminated**

---

## Implementation Plan

### Part 1: Modify ClassicEvalRunResult Data Structure

**File**: `src/scorebook/types.py`

#### 1.1 Change scores Field Type

**Current** (line 63):
```python
@dataclass
class ClassicEvalRunResult:
    run_spec: EvalRunSpec
    run_completed: bool
    outputs: Optional[List[Any]]
    scores: Optional[Dict[str, Any]]  # Metric-centric format
    run_id: Optional[str] = None
```

Current scores format:
```python
{
    "accuracy": {
        "aggregate_scores": {"accuracy": 0.95},
        "item_scores": [1, 0, 1, ...]
    },
    "f1": {
        "aggregate_scores": {"f1": 0.92, "precision": 0.90},
        "item_scores": [0.95, 0.88, ...]
    }
}
```

**New**:
```python
@dataclass
class ClassicEvalRunResult:
    run_spec: EvalRunSpec
    run_completed: bool
    outputs: Optional[List[Any]]
    scores: Optional[Dict[str, List[Dict[str, Any]]]]  # score_async format
    run_id: Optional[str] = None
```

New scores format (from score_async):
```python
{
    "aggregate_results": [
        {
            "dataset": "my_dataset",
            "temperature": 0.7,  # hyperparameters flattened
            "accuracy": 0.95,
            "f1": 0.92,
            "f1_precision": 0.90,
            "f1_recall": 0.94,
            "run_id": "abc123"  # if upload succeeded
        }
    ],
    "item_results": [
        {
            "id": 0,
            "dataset_name": "my_dataset",
            "input": "question text",
            "output": "model answer",
            "label": "correct answer",
            "temperature": 0.7,  # hyperparameters
            "accuracy": 1,
            "f1": 0.95,
            "run_id": "abc123"  # if upload succeeded
        },
        ...
    ]
}
```

**Rationale**: This format is the natural output of score_async() and matches what users expect from score(). It's result-centric rather than metric-centric.

#### 1.2 Update item_scores Property

**Current** (lines 67-101 in types.py):
```python
@property
def item_scores(self) -> List[Dict[str, Any]]:
    """Return a list of dictionaries containing scores for each evaluated item."""
    results = []

    if self.outputs:
        for idx, output in enumerate(self.outputs):
            if idx >= len(self.run_spec.inputs):
                break

            result = {
                "id": idx,
                "dataset_name": self.run_spec.dataset.name,
                "input": self.run_spec.inputs[idx],
                "label": self.run_spec.labels[idx] if idx < len(self.run_spec.labels) else None,
                "output": output,
                **self.run_spec.hyperparameter_config,
            }

            if self.run_id is not None:
                result["run_id"] = self.run_id

            # Extract item scores from metric-centric format
            if self.scores is not None:
                for metric_name, metric_data in self.scores.items():
                    if isinstance(metric_data, dict) and "item_scores" in metric_data:
                        if idx < len(metric_data["item_scores"]):
                            result[metric_name] = metric_data["item_scores"][idx]
                    else:
                        result[metric_name] = metric_data

            results.append(result)

    return results
```

**New** (much simpler):
```python
@property
def item_scores(self) -> List[Dict[str, Any]]:
    """Return a list of dictionaries containing scores for each evaluated item."""
    if self.scores and "item_results" in self.scores:
        # score_async already built this in the exact format we need
        return self.scores["item_results"]
    return []
```

**Reduction**: 30 lines → 4 lines

#### 1.3 Update aggregate_scores Property

**Current** (lines 104-128 in types.py):
```python
@property
def aggregate_scores(self) -> Dict[str, Any]:
    """Return the aggregated scores for this run."""
    result = {
        "dataset": self.run_spec.dataset.name,
        "run_completed": self.run_completed,
        **self.run_spec.hyperparameter_config,
    }

    if self.run_id is not None:
        result["run_id"] = self.run_id

    # Extract aggregate scores from metric-centric format
    if self.scores is not None:
        for metric_name, metric_data in self.scores.items():
            if isinstance(metric_data, dict) and "aggregate_scores" in metric_data:
                for key, value in metric_data["aggregate_scores"].items():
                    score_key = key if key == metric_name else f"{metric_name}_{key}"
                    result[score_key] = value
            else:
                result[metric_name] = metric_data

    return result
```

**New** (much simpler):
```python
@property
def aggregate_scores(self) -> Dict[str, Any]:
    """Return the aggregated scores for this run."""
    if self.scores and "aggregate_results" in self.scores and len(self.scores["aggregate_results"]) > 0:
        result = self.scores["aggregate_results"][0].copy()
        # Add run_completed (not included in score_async format)
        result["run_completed"] = self.run_completed
        return result

    # Fallback if no scores available
    return {
        "dataset": self.run_spec.dataset.name,
        "run_completed": self.run_completed,
        **self.run_spec.hyperparameter_config,
    }
```

**Reduction**: 25 lines → 12 lines

**Note**: score_async() already includes dataset, hyperparameters, metrics, and run_id in aggregate_results. We just need to add run_completed.

---

### Part 2: Refactor execute_classic_eval_run() to Use score_async()

**File**: `src/scorebook/evaluate/_async/evaluate_async.py`

#### 2.1 Add Imports

Add this import at the top:
```python
from scorebook.score._async.score_async import score_async
```

Remove (no longer needed):
```python
from scorebook.evaluate.evaluate_helpers import score_metrics  # DELETE
```

#### 2.2 Update Function Signature

**Current** (line 237):
```python
async def execute_classic_eval_run(
    inference: Callable,
    run: EvalRunSpec
) -> ClassicEvalRunResult:
```

**New**:
```python
async def execute_classic_eval_run(
    inference: Callable,
    run: EvalRunSpec,
    upload_results: bool,
    experiment_id: Optional[str],
    project_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> ClassicEvalRunResult:
```

**Rationale**: Need these parameters to pass to score_async().

#### 2.3 Rewrite Function Body

**Current** (lines 237-254):
```python
async def execute_classic_eval_run(inference: Callable, run: EvalRunSpec) -> ClassicEvalRunResult:
    """Execute a classic evaluation run."""
    logger.debug("Executing classic eval run for %s", run)

    inference_outputs = None
    metric_scores = None

    try:
        inference_outputs = await run_inference_callable(
            inference, run.inputs, run.hyperparameter_config
        )
        metric_scores = score_metrics(run.dataset, inference_outputs, run.labels)
        logger.debug("Classic evaluation completed for run %s", run)
        return ClassicEvalRunResult(run, True, inference_outputs, metric_scores)

    except Exception as e:
        logger.warning("Failed to complete classic eval run for %s: %s", run, str(e))
        return ClassicEvalRunResult(run, False, inference_outputs, metric_scores)
```

**New**:
```python
async def execute_classic_eval_run(
    inference: Callable,
    run: EvalRunSpec,
    upload_results: bool,
    experiment_id: Optional[str],
    project_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> ClassicEvalRunResult:
    """Execute a classic evaluation run using score_async() for scoring and uploading."""
    logger.debug("Executing classic eval run for %s", run)

    inference_outputs = None
    scores = None

    try:
        # 1. Run inference
        inference_outputs = await run_inference_callable(
            inference, run.inputs, run.hyperparameter_config
        )

        # 2. Build items for score_async
        items = [
            {
                "input": run.inputs[i] if i < len(run.inputs) else None,
                "output": inference_outputs[i],
                "label": run.labels[i] if i < len(run.labels) else "",
            }
            for i in range(len(inference_outputs))
        ]

        # 3. Get model name for upload
        model_name = get_model_name(inference, metadata)

        # 4. Call score_async (handles both scoring AND uploading)
        scores = await score_async(
            items=items,
            metrics=run.dataset.metrics,
            output="output",  # Explicit parameter
            label="label",    # Explicit parameter
            input="input",    # Explicit parameter
            hyperparameters=run.hyperparameter_config,
            dataset_name=run.dataset.name,
            model_name=model_name,
            metadata=metadata,
            experiment_id=experiment_id,
            project_id=project_id,
            upload_results=upload_results,
        )

        # 5. Extract run_id if upload succeeded
        run_id = None
        if scores.get("aggregate_results") and len(scores["aggregate_results"]) > 0:
            run_id = scores["aggregate_results"][0].get("run_id")

        logger.debug("Classic evaluation completed for run %s (run_id: %s)", run, run_id)
        return ClassicEvalRunResult(
            run_spec=run,
            run_completed=True,
            outputs=inference_outputs,
            scores=scores,
            run_id=run_id,
        )

    except Exception as e:
        logger.warning("Failed to complete classic eval run for %s: %s", run, str(e))
        return ClassicEvalRunResult(
            run_spec=run,
            run_completed=False,
            outputs=inference_outputs,
            scores=scores,
            run_id=None,
        )
```

**Key Changes**:
- Builds items list from inputs/outputs/labels
- Calls score_async() instead of score_metrics()
- score_async() handles upload internally (no separate upload step!)
- Extracts run_id from score_async() results
- Stores full score_async() dict in scores field
- Explicit output/label/input parameters for clarity

---

### Part 3: Update execute_run() to Pass Parameters

**File**: `src/scorebook/evaluate/_async/evaluate_async.py`

**Current** (lines 208-219):
```python
async def execute_run(
    inference: Callable,
    run: Union[EvalRunSpec, AdaptiveEvalRunSpec],
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trismik_client: Optional[Union[TrismikClient, TrismikAsyncClient]] = None,
) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
    """Execute a single evaluation run."""

    if isinstance(run, EvalRunSpec):
        return await execute_classic_eval_run(inference, run)
```

**New**:
```python
async def execute_run(
    inference: Callable,
    run: Union[EvalRunSpec, AdaptiveEvalRunSpec],
    upload_results: bool,  # NEW PARAMETER
    experiment_id: Optional[str] = None,
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trismik_client: Optional[Union[TrismikClient, TrismikAsyncClient]] = None,
) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
    """Execute a single evaluation run."""

    if isinstance(run, EvalRunSpec):
        return await execute_classic_eval_run(
            inference, run, upload_results, experiment_id, project_id, metadata
        )
```

**Note**: trismik_client still needed for adaptive evals, but not used for classic evals anymore (score_async creates its own client via upload_run_result_async).

---

### Part 4: Update Worker in execute_runs()

**File**: `src/scorebook/evaluate/_async/evaluate_async.py`

**Current** (lines 157-194):
```python
async def worker(
    run: Union[EvalRunSpec, AdaptiveEvalRunSpec]
) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
    run_result = await execute_run(
        inference, run, experiment_id, project_id, metadata, trismik_client
    )
    # Update progress bars with items processed and success status
    if progress_bars is not None:
        items_processed = (
            len(run.dataset.items)
            if isinstance(run, EvalRunSpec)
            else evaluation_settings["max_iterations"]
        )
        progress_bars.on_run_completed(items_processed, run_result.run_completed)

    # UPLOAD LOGIC - TO BE REMOVED
    if (
        upload_results
        and isinstance(run_result, ClassicEvalRunResult)
        and experiment_id
        and project_id
        and run_result.run_completed
        and trismik_client is not None
    ):
        try:
            run_id = await upload_classic_run_results(
                run_result, experiment_id, project_id, inference, metadata, trismik_client
            )
            run_result.run_id = run_id
            if progress_bars is not None:
                progress_bars.on_upload_completed(succeeded=True)
        except Exception as e:
            logger.warning(f"Failed to upload run results: {e}")
            if progress_bars is not None:
                progress_bars.on_upload_completed(succeeded=False)
            # Continue evaluation even if upload fails

    return run_result
```

**New** (much simpler):
```python
async def worker(
    run: Union[EvalRunSpec, AdaptiveEvalRunSpec]
) -> Union[ClassicEvalRunResult, AdaptiveEvalRunResult]:
    # Execute run (score_async handles upload internally for classic evals)
    run_result = await execute_run(
        inference, run, upload_results, experiment_id, project_id, metadata, trismik_client
    )

    # Update progress bars with items processed and success status
    if progress_bars is not None:
        items_processed = (
            len(run.dataset.items)
            if isinstance(run, EvalRunSpec)
            else evaluation_settings["max_iterations"]
        )
        progress_bars.on_run_completed(items_processed, run_result.run_completed)

    # Update upload progress for classic evals (upload already happened in score_async)
    if upload_results and isinstance(run_result, ClassicEvalRunResult) and run_result.run_completed:
        # Check if upload succeeded by checking for run_id
        if experiment_id and project_id:
            upload_succeeded = run_result.run_id is not None
            if progress_bars is not None:
                progress_bars.on_upload_completed(succeeded=upload_succeeded)

    return run_result
```

**Key Changes**:
- Pass upload_results to execute_run
- Remove entire upload_classic_run_results() call (score_async already did it!)
- Update upload progress based on presence of run_id
- Much simpler logic: ~37 lines → ~24 lines

**Note on upload timing**: score_async() catches upload exceptions internally (doesn't raise), so run_result.run_completed=True even if upload fails. We detect upload failure by checking if run_id is None.

---

### Part 5: Delete upload_classic_run_results() Entirely

**File**: `src/scorebook/evaluate/_async/evaluate_async.py`

**DELETE** lines 316-408 (the entire upload_classic_run_results function):

```python
async def upload_classic_run_results(  # DELETE THIS ENTIRE FUNCTION
    run_result: ClassicEvalRunResult,
    experiment_id: str,
    project_id: str,
    inference_callable: Optional[Callable],
    metadata: Optional[Dict[str, Any]],
    trismik_client: Union[TrismikClient, TrismikAsyncClient],
) -> str:
    # ... 93 lines of duplicated logic ...
```

**Result**: Eliminates 93 lines of code!

---

### Part 5.5: Update Test Files

Tests currently access `.scores` directly and expect the old metric-centric format. They need to be updated to use the public API (properties).

#### File: `tests/test_evaluate/test_evaluate.py`

**Lines 419-420** (and similar at lines 493-494):
```python
# OLD
assert run.scores is not None
assert "accuracy" in run.scores

# NEW - use public API (properties)
assert run.scores is not None
assert "accuracy" in run.aggregate_scores
```

**Lines 426, 500** (no change needed):
```python
assert run.scores is None  # Still works for failed runs ✅
```

#### File: `tests/test_evaluate/test_evaluate_async.py`

**Line 101**:
```python
# OLD
assert sync_eval.scores == async_eval.scores

# NEW - compare through public API
assert sync_eval.aggregate_scores == async_eval.aggregate_scores
assert sync_eval.item_scores == async_eval.item_scores
```

**Rationale**: Tests should use the public API (properties) rather than internal structure (.scores). The properties maintain backward compatibility while the internal structure has changed.

---

### Part 6: Auto-Generate Sync Version

The sync version (evaluate.py) is automatically generated using `unasync`.

**Steps**:

1. Complete all changes to `src/scorebook/evaluate/_async/evaluate_async.py`

2. Run the generator:
   ```bash
   python scripts/generate_sync.py
   ```

3. Verify the output:
   ```bash
   git diff src/scorebook/evaluate/_sync/evaluate.py
   ```

4. The generator automatically applies these transformations:
   - `score_async` → `score`
   - `async def` → `def`
   - `await` → (removed)
   - `upload_run_result_async` → `upload_run_result`
   - `create_trismik_async_client` → `create_trismik_sync_client`
   - And other replacements from `pyproject.toml`

**No manual edits needed!** All transformations are configured in `pyproject.toml` under `[[tool.unasync.rules]]`.

**Configuration** (already in pyproject.toml):
```toml
[[tool.unasync.rules]]
fromdir = "src/scorebook/evaluate/_async/"
todir = "src/scorebook/evaluate/_sync/"
replacements.evaluate_async = "evaluate"
replacements.create_trismik_async_client = "create_trismik_sync_client"
replacements.async_nullcontext = "nullcontext"

[[tool.unasync.rules]]
fromdir = "src/scorebook/score/_async/"
todir = "src/scorebook/score/_sync/"
replacements.score_async = "score"
replacements.upload_run_result_async = "upload_run_result"
```

---

## Summary of Changes

### Files Modified

1. **src/scorebook/types.py**
   - Change ClassicEvalRunResult.scores type
   - Simplify item_scores property (30 lines → 4 lines)
   - Simplify aggregate_scores property (25 lines → 12 lines)

2. **src/scorebook/evaluate/_async/evaluate_async.py**
   - Add import for score_async
   - Remove import for score_metrics
   - Update execute_classic_eval_run signature (add parameters)
   - Rewrite execute_classic_eval_run to call score_async
   - Update execute_run signature (add upload_results parameter)
   - Simplify worker function (remove upload logic)
   - DELETE upload_classic_run_results function (93 lines)

3. **tests/test_evaluate/test_evaluate.py**
   - Update to use properties instead of direct .scores access

4. **tests/test_evaluate/test_evaluate_async.py**
   - Update score comparison to use properties

5. **src/scorebook/evaluate/_sync/evaluate.py**
   - Auto-generated (no manual edits)

### Code Reduction

- ClassicEvalRunResult properties: **43 lines → 16 lines** (-27 lines)
- execute_classic_eval_run: More explicit but delegates to score_async
- worker function: **37 lines → 24 lines** (-13 lines)
- upload_classic_run_results: **93 lines → 0 lines** (-93 lines)

**Total Reduction: ~133 lines of code + eliminates all duplication**

---

## Testing Strategy

### Unit Tests

1. **Test ClassicEvalRunResult properties**:
   ```python
   # Verify item_scores returns score_async format correctly
   def test_item_scores_with_new_format():
       scores = {
           "aggregate_results": [{"dataset": "test", "accuracy": 0.95}],
           "item_results": [{"id": 0, "accuracy": 1, ...}, ...]
       }
       result = ClassicEvalRunResult(run_spec, True, outputs, scores, "run123")
       assert result.item_scores == scores["item_results"]

   # Verify aggregate_scores adds run_completed
   def test_aggregate_scores_with_new_format():
       scores = {
           "aggregate_results": [{"dataset": "test", "accuracy": 0.95}],
           "item_results": [...]
       }
       result = ClassicEvalRunResult(run_spec, True, outputs, scores, None)
       assert result.aggregate_scores["run_completed"] == True
       assert result.aggregate_scores["accuracy"] == 0.95
   ```

2. **Test execute_classic_eval_run**:
   ```python
   # Verify it calls score_async with correct parameters
   @patch('scorebook.evaluate._async.evaluate_async.score_async')
   async def test_execute_classic_eval_run_calls_score_async(mock_score):
       mock_score.return_value = {"aggregate_results": [...], "item_results": [...]}

       result = await execute_classic_eval_run(
           inference, run, upload_results=True,
           experiment_id="exp1", project_id="proj1", metadata={}
       )

       # Verify score_async was called with correct items
       assert mock_score.called
       call_args = mock_score.call_args
       assert call_args.kwargs["upload_results"] == True
       assert call_args.kwargs["experiment_id"] == "exp1"
   ```

3. **Test worker upload progress**:
   ```python
   # Verify upload progress updated based on run_id presence
   async def test_worker_updates_upload_progress():
       # When upload succeeds (run_id present)
       result_with_id = ClassicEvalRunResult(..., run_id="abc123")
       # Should call progress_bars.on_upload_completed(succeeded=True)

       # When upload fails (no run_id)
       result_without_id = ClassicEvalRunResult(..., run_id=None)
       # Should call progress_bars.on_upload_completed(succeeded=False)
   ```

### Integration Tests

1. **End-to-end evaluate_async with upload**:
   ```python
   async def test_evaluate_async_with_upload():
       result = await evaluate_async(
           inference=my_inference,
           datasets=dataset,
           upload_results=True,
           experiment_id="exp1",
           project_id="proj1",
       )

       # Verify results have run_id
       assert result["aggregate_results"][0]["run_id"] is not None
       assert all(item["run_id"] for item in result["item_results"])
   ```

2. **Verify uploaded data matches previous implementation**:
   - Compare Trismik API requests before/after refactor
   - Ensure same items/metrics are uploaded

3. **Test error handling**:
   - Upload failures should not break evaluation
   - Progress bars should indicate failure
   - run_id should be None when upload fails

### Regression Tests

Run existing evaluate() test suite to ensure no breaking changes:
- Multiple datasets
- Multiple hyperparameter configs
- With/without upload
- Adaptive evals still work
- Progress bars work correctly

---

## Migration Risks & Mitigation

### Risk 1: ClassicEvalRunResult.scores format change

**Impact**: Code that directly accesses `result.scores` expecting old format will break.

**Who is affected**:
- ❌ Internal tests (need updates - see Part 5.5)
- ❌ Advanced users who access .scores directly (very rare)
- ✅ Default users (return_dict=True) - NOT affected (95%+ of users)
- ✅ Users who access properties - NOT affected

**Scope**: Very limited! ~95% of users unaffected because:
1. Default return type is dict (via properties)
2. Public API is through properties (item_scores, aggregate_scores)
3. ClassicEvalRunResult is internal (not in public __init__.py)

**Mitigation**:
- Update internal tests (Part 5.5)
- Add migration notes for advanced users
- Properties maintain same return format
- Search for direct access in your code:
  ```bash
  git grep "\.scores\[" | grep -v aggregate_scores | grep -v item_scores
  ```

### Risk 2: Different metric naming

**Impact**: score_async() flattens metric names (e.g., "f1_precision" vs accessing nested dict).

**Mitigation**:
- Both formats use same flattening logic (from format_results in score_helpers.py)
- Test with multi-aggregate metrics (like F1)
- Verify uploaded data has correct metric names

### Risk 3: Upload timing and progress bars

**Impact**: Upload now happens inside execute_run instead of worker.

**Mitigation**:
- Worker still updates progress based on run_id presence
- Test all progress bar states
- Ensure upload failures are detected and reported
- score_async() catches upload exceptions internally (doesn't raise)

### Risk 4: trismik_client no longer needed for classic evals

**Impact**: score_async creates its own client, making passed client redundant.

**Mitigation**:
- Keep parameter for adaptive evals
- Document that it's ignored for classic evals
- upload_run_result_async() creates client internally (line 209 in upload_results.py)

---

## Implementation Checklist

### Phase 1: Update Data Structures
- [ ] Modify ClassicEvalRunResult.scores type in types.py
- [ ] Update item_scores property (30 lines → 4 lines)
- [ ] Update aggregate_scores property (25 lines → 12 lines)
- [ ] Run unit tests for ClassicEvalRunResult

### Phase 2: Refactor evaluate_async.py
- [ ] Add score_async import
- [ ] Remove score_metrics import
- [ ] Update execute_classic_eval_run signature
- [ ] Rewrite execute_classic_eval_run body
- [ ] Update execute_run signature
- [ ] Pass upload_results through execute_run
- [ ] Simplify worker function
- [ ] Remove upload_classic_run_results function (DELETE 93 lines)
- [ ] Update test_evaluate.py (Part 5.5)
- [ ] Update test_evaluate_async.py (Part 5.5)

### Phase 3: Generate Sync Version
- [ ] Run `python scripts/generate_sync.py`
- [ ] Verify generated evaluate.py
- [ ] Do NOT manually edit evaluate.py
- [ ] Check git diff for expected changes

### Phase 4: Testing
- [ ] Run unit tests
- [ ] Run integration tests
- [ ] Test with real Trismik uploads (if available)
- [ ] Verify progress bars work correctly
- [ ] Test error handling (upload failures)
- [ ] Run full test suite

### Phase 5: Cleanup & Documentation
- [ ] Remove unused imports
- [ ] Update CHANGELOG with migration notes
- [ ] Add internal documentation comments
- [ ] Consider deprecation notice for score_metrics in evaluate_helpers

---

## Success Criteria

1. ✅ All existing tests pass (after test updates)
2. ✅ ClassicEvalRunResult properties return same format as before
3. ✅ evaluate() and score() share same scoring/upload implementation
4. ✅ Uploads work identically to before
5. ✅ Progress bars update correctly
6. ✅ ~133 lines of code removed
7. ✅ No code duplication between evaluate and score
8. ✅ Error handling works (upload failures don't break eval)
9. ✅ Sync version auto-generated correctly
10. ✅ Public API unchanged (return_dict=True users see no change)

---

## Long-term Benefits

1. **Single Source of Truth**: All scoring + upload logic in score_async()
2. **Automatic Improvements**: Future score_async() enhancements automatically benefit evaluate()
3. **Simpler Codebase**: Less code to maintain and test
4. **Better API Consistency**: evaluate() and score() work the same way
5. **Easier Debugging**: Only one place to debug scoring/upload issues
6. **Future Extensibility**: Easy to add new features to score_async() knowing they'll work in evaluate()
7. **No Sync/Async Maintenance**: unasync automatically keeps them in sync

---

## Conclusion

This refactor achieves the **maximum possible code reuse** between evaluate() and score() by making evaluate_async() call score_async() directly for classic evaluations.

**Key advantages over original plan**:
- ✅ Correctly uses unasync for sync generation (not manual edits)
- ✅ Includes necessary test updates
- ✅ Clarifies limited breaking change scope
- ✅ More accurate code reduction estimate

The data structure change to ClassicEvalRunResult is minimal and internal-only (properties maintain the same interface), while the code reduction and simplification are substantial (~133 lines removed + all duplication eliminated).

**This is the cleanest long-term solution that makes evaluate() and score() true partners in the codebase.**
