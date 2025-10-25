# Refactor Plan Review & Analysis

## Executive Summary

✅ **The plan is fundamentally sound and will achieve the stated goals**

However, there are **critical corrections and additions** needed:

1. **Sync version generation** - Plan incorrectly suggests manual sync updates
2. **Test updates required** - Tests access `.scores` directly and will break
3. **Breaking change scope** - More limited than initially thought
4. **Progress bar timing** - Upload happens earlier in the flow than planned

## Detailed Analysis

### ✅ CORRECT: Core Strategy

The plan correctly identifies that:
- score_async() already does scoring + uploading
- evaluate_async() duplicates this logic (~115 lines)
- Calling score_async() directly eliminates duplication
- ClassicEvalRunResult.scores should store score_async() format

### ❌ CRITICAL ISSUE #1: Sync Version Generation

**Plan states** (Part 6):
> Apply the same changes but with sync equivalents:
> 1. Import `score` instead of `score_async`
> 2. Remove `async`/`await` keywords
> ...

**Reality**:
The sync version is **automatically generated** by `scripts/generate_sync.py` using the `unasync` library.

**Configuration** (`pyproject.toml`):
```toml
[[tool.unasync.rules]]
fromdir = "src/scorebook/evaluate/_async/"
todir = "src/scorebook/evaluate/_sync/"
replacements.evaluate_async = "evaluate"
replacements.create_trismik_async_client = "create_trismik_sync_client"
replacements.async_nullcontext = "nullcontext"
...

[[tool.unasync.rules]]
fromdir = "src/scorebook/score/_async/"
todir = "src/scorebook/score/_sync/"
replacements.score_async = "score"
replacements.calculate_metric_scores_async = "calculate_metric_scores"
replacements.upload_run_result_async = "upload_run_result"
```

**Impact**:
- We ONLY modify `evaluate_async.py`
- Run `python scripts/generate_sync.py` to auto-generate `evaluate.py`
- No manual sync version edits needed!

**Required Changes to Plan**:

1. **Delete Part 6** entirely
2. **Add new section**:

```markdown
### Part 6: Generate Sync Version

**After completing all changes to evaluate_async.py**:

1. Run the sync generator:
   ```bash
   python scripts/generate_sync.py
   ```

2. This automatically:
   - Converts `score_async` → `score`
   - Converts `await` → (removed)
   - Converts `async def` → `def`
   - Converts `upload_run_result_async` → `upload_run_result`
   - Generates `src/scorebook/evaluate/_sync/evaluate.py`

3. **No manual edits needed** - unasync handles everything!

4. Verify generated file:
   ```bash
   git diff src/scorebook/evaluate/_sync/evaluate.py
   ```

**Note**: The unasync config in `pyproject.toml` already has all needed replacements.
```

---

### ❌ CRITICAL ISSUE #2: Test Updates Required

**Plan mentions** (Migration Risks):
> Search codebase for direct `.scores` access

**Reality**:
Tests **DO** access `.scores` directly and expect the old format:

**Found in tests**:
```python
# test_evaluate.py:419-420
assert run.scores is not None
assert "accuracy" in run.scores  # ❌ WILL BREAK

# test_evaluate_async.py:101
assert sync_eval.scores == async_eval.scores  # ❌ WILL BREAK
```

**Why it breaks**:
```python
# Old format (expected by tests):
run.scores = {
    "accuracy": {"aggregate_scores": {...}, "item_scores": [...]}
}
# "accuracy" in run.scores → True ✅

# New format (after refactor):
run.scores = {
    "aggregate_results": [{...}],
    "item_results": [...]
}
# "accuracy" in run.scores → False ❌
```

**Required Changes to Plan**:

Add new section after Part 5:

```markdown
### Part 5.5: Update Tests

**Files to update**:
- `tests/test_evaluate/test_evaluate.py`
- `tests/test_evaluate/test_evaluate_async.py`

**Changes needed**:

1. **Replace direct .scores checks**:
   ```python
   # OLD
   assert "accuracy" in run.scores

   # NEW
   assert run.scores is not None
   assert run.scores.get("aggregate_results")
   assert any("accuracy" in agg for agg in run.scores["aggregate_results"])

   # OR (better - use properties)
   assert "accuracy" in run.aggregate_scores
   ```

2. **Update score comparison tests**:
   ```python
   # OLD
   assert sync_eval.scores == async_eval.scores

   # NEW - compare through properties (public API)
   assert sync_eval.aggregate_scores == async_eval.aggregate_scores
   assert sync_eval.item_scores == async_eval.item_scores
   ```

3. **Verify None checks still work**:
   ```python
   assert run.scores is None  # Still works for failed runs ✅
   ```

**Alternative approach**: Update tests to ONLY use properties (recommended):
```python
# Don't access .scores directly - use public properties
assert "accuracy" in run.aggregate_scores  # ✅ Public API
assert len(run.item_scores) > 0  # ✅ Public API
```
```

---

### ⚠️ ISSUE #3: Upload Progress Bar Timing

**Plan states** (Part 4):
> Update upload progress for classic evals (upload already happened in score_async)

**Analysis**:

Current flow:
```python
worker():
  run_result = execute_run()  # scoring happens here
  progress_bars.on_run_completed()
  # Upload happens here (AFTER on_run_completed)
  upload_classic_run_results()
  progress_bars.on_upload_completed()
```

Planned flow:
```python
worker():
  run_result = execute_run()
    ↳ execute_classic_eval_run()
      ↳ score_async()  # Upload happens HERE
  progress_bars.on_run_completed()
  # Check if upload succeeded based on run_id
  progress_bars.on_upload_completed()
```

**Issue**: `on_upload_completed()` is called AFTER `on_run_completed()` in both cases, so this is actually fine!

**However**, there's a subtle issue: If score_async() fails during upload, the exception is caught internally (line 98-100 of score_async.py):

```python
except Exception as e:
    logger.warning(f"Failed to upload score results: {e}")
    # Don't raise - continue execution even if upload fails
```

So the run completes successfully but upload fails. The plan correctly handles this by checking for run_id presence.

**Verdict**: ✅ Plan is correct here, but should clarify the exception handling

---

### ✅ CORRECT: Data Structure Changes

The plan correctly identifies:

1. **ClassicEvalRunResult.scores type change**:
   ```python
   # Old: Optional[Dict[str, Any]]
   # New: Optional[Dict[str, List[Dict[str, Any]]]]
   ```

2. **Property simplifications**:
   - `item_scores`: 30 lines → 4 lines ✅
   - `aggregate_scores`: 25 lines → 12 lines ✅

3. **Properties maintain public API** ✅

**Verification**:
```python
# Both old and new return same format:
result.item_scores → List[Dict[str, Any]]  # ✅ Same
result.aggregate_scores → Dict[str, Any]   # ✅ Same
```

---

### ✅ CORRECT: Function Signatures

Plan correctly updates:

1. **execute_classic_eval_run** signature to accept:
   - upload_results
   - experiment_id
   - project_id
   - metadata

2. **execute_run** signature to accept:
   - upload_results (new parameter)

3. **worker** function simplified

---

### ⚠️ ISSUE #4: Breaking Change Scope

**Plan states**:
> May break existing code that accesses .scores directly

**Reality** (after investigation):

**Who is affected**:

1. ✅ **Default users (return_dict=True)** - NOT affected
   ```python
   results = evaluate(...)  # Returns dict via properties
   # No change for them!
   ```

2. ✅ **Property users (return_dict=False)** - NOT affected
   ```python
   eval_result = evaluate(..., return_dict=False)
   eval_result.aggregate_scores  # Still works
   eval_result.item_scores  # Still works
   ```

3. ❌ **Advanced users** - AFFECTED (small number)
   ```python
   eval_result = evaluate(..., return_dict=False)
   eval_result.run_results[0].scores  # Direct access - breaks!
   ```

4. ❌ **Internal tests** - AFFECTED
   ```python
   assert "accuracy" in run.scores  # Breaks!
   ```

**Impact**: Much smaller than initially thought!

**Recommendation**:
- Document as internal API change
- Update migration guide with examples
- Most users won't notice

---

### ✅ CORRECT: Code Reduction

Plan correctly calculates:

- Properties: -43 lines
- worker: -17 lines
- upload_classic_run_results: -93 lines (deleted)
- **Total: ~137 lines removed** ✅

Plus eliminates ~115 lines of duplicated logic ✅

---

### ⚠️ ISSUE #5: Missing Implementation Detail

**Gap in plan**: How to handle the `input` field in score_async()

**Current score_async signature**:
```python
async def score_async(
    items: List[Dict[str, Any]],
    metrics: Metrics,
    output: str = "output",  # Default key name
    label: str = "label",    # Default key name
    input: Optional[str] = None,  # Optional key name
    ...
)
```

**Plan shows** (Part 2.3):
```python
items = [
    {
        "input": run.inputs[i] if i < len(run.inputs) else None,
        "output": inference_outputs[i],
        "label": run.labels[i] if i < len(run.labels) else "",
    }
    for i in range(len(inference_outputs))
]
```

**Issue**: We're hardcoding "input" as the key, but score_async has `input` as an optional parameter.

**Solution**: Call score_async with explicit input parameter:
```python
scores = await score_async(
    items=items,
    metrics=run.dataset.metrics,
    output="output",  # Explicit
    label="label",    # Explicit
    input="input",    # Explicit
    hyperparameters=run.hyperparameter_config,
    ...
)
```

This is minor but should be in the plan for clarity.

---

## Summary of Required Plan Updates

### Critical Updates

1. ✅ **Replace Part 6** with auto-generation instructions
2. ✅ **Add Part 5.5** for test updates
3. ✅ **Update Migration Risks** with actual breaking change scope
4. ✅ **Add test updates to Implementation Checklist**

### Minor Updates

5. ⚠️ **Clarify exception handling** in upload flow
6. ⚠️ **Add explicit input parameter** in score_async call
7. ⚠️ **Add note about unasync** in introduction

---

## Overall Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Core Strategy | ✅ Excellent | Correctly identifies duplication and solution |
| Data Structure Changes | ✅ Correct | Type changes and property updates are accurate |
| Code Reduction | ✅ Accurate | ~137 lines + eliminates duplication |
| Breaking Change Analysis | ⚠️ Overstated | Much smaller scope than initially thought |
| Sync Version Handling | ❌ Incorrect | Must use unasync, not manual edits |
| Test Impact | ❌ Missing | Tests need updates, not mentioned in checklist |
| Implementation Details | ✅ Mostly Complete | Minor gaps in score_async call |

**Overall Grade: B+** (A- after corrections)

---

## Recommended Corrections

### 1. Update Part 6

**Delete current Part 6 and replace with**:

```markdown
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
   - And other replacements from `pyproject.toml`

**No manual edits needed!** All transformations are configured in `pyproject.toml` under `[[tool.unasync.rules]]`.
```

### 2. Add Part 5.5 (Test Updates)

Insert between Part 5 and Part 6:

```markdown
### Part 5.5: Update Test Files

**File**: `tests/test_evaluate/test_evaluate.py`

**Changes needed**:

Lines 419-420:
```python
# OLD
assert run.scores is not None
assert "accuracy" in run.scores

# NEW - use public API (properties)
assert run.scores is not None
assert "accuracy" in run.aggregate_scores
```

Lines 493-494 (similar pattern):
```python
# OLD
assert run.scores is not None
assert "accuracy" in run.scores

# NEW
assert run.scores is not None
assert "accuracy" in run.aggregate_scores
```

Lines 426, 500:
```python
assert run.scores is None  # No change needed - still works ✅
```

**File**: `tests/test_evaluate/test_evaluate_async.py`

Line 101:
```python
# OLD
assert sync_eval.scores == async_eval.scores

# NEW - compare through public API
assert sync_eval.aggregate_scores == async_eval.aggregate_scores
assert sync_eval.item_scores == async_eval.item_scores
```

**Rationale**: Tests should use public API (properties) not internal structure (.scores).
```

### 3. Update Implementation Checklist

Add to Phase 2:
```markdown
- [ ] Update test_evaluate.py (Part 5.5)
- [ ] Update test_evaluate_async.py (Part 5.5)
```

Add to Phase 3:
```markdown
- [ ] Run python scripts/generate_sync.py
- [ ] Verify generated evaluate.py
- [ ] Do NOT manually edit evaluate.py
```

### 4. Update Migration Risks Section

Replace "Risk 1" with:

```markdown
### Risk 1: ClassicEvalRunResult.scores format change

**Impact**: Code that directly accesses `result.scores` expecting old format will break.

**Who is affected**:
- ❌ Internal tests (need updates - see Part 5.5)
- ❌ Advanced users who access .scores directly (very rare)
- ✅ Default users (return_dict=True) - NOT affected
- ✅ Users who access properties - NOT affected

**Scope**: Very limited! ~95% of users unaffected.

**Mitigation**:
- ClassicEvalRunResult is internal (not in public __init__.py)
- Public interface through properties unchanged
- Update internal tests (Part 5.5)
- Add migration notes for advanced users
- Search for direct access:
  ```bash
  git grep "\.scores\[" | grep -v aggregate_scores | grep -v item_scores
  ```
```

### 5. Add Unasync Note to Introduction

Add to "Critical Insight" section:

```markdown
**Important**: The sync version (`evaluate.py`) is auto-generated from the async version using `unasync`. We only modify `evaluate_async.py` and run `python scripts/generate_sync.py` to generate `evaluate.py`.
```

---

## Additional Recommendations

### Recommendation 1: Add Explicit Parameters

In Part 2.3, update the score_async call:

```python
scores = await score_async(
    items=items,
    metrics=run.dataset.metrics,
    output="output",        # ADD: Explicit parameter
    label="label",          # ADD: Explicit parameter
    input="input",          # ADD: Explicit parameter
    hyperparameters=run.hyperparameter_config,
    dataset_name=run.dataset.name,
    model_name=model_name,
    metadata=metadata,
    experiment_id=experiment_id,
    project_id=project_id,
    upload_results=upload_results,
)
```

### Recommendation 2: Add Debug Logging

Consider adding debug logging in execute_classic_eval_run:

```python
logger.debug(f"Calling score_async with {len(items)} items")
scores = await score_async(...)
logger.debug(f"score_async returned with run_id: {run_id}")
```

This helps debug upload success/failure.

### Recommendation 3: Add Validation

After score_async call, validate the format:

```python
# Validate score_async returned expected format
if not isinstance(scores, dict):
    raise ScoreBookError(f"score_async returned unexpected type: {type(scores)}")
if "aggregate_results" not in scores or "item_results" not in scores:
    raise ScoreBookError("score_async returned invalid format")
```

---

## Conclusion

**The refactor plan is fundamentally sound and achieves the stated goals.**

With the corrections above:
- ✅ Eliminates ~137 lines of code
- ✅ Removes all duplication between evaluate and score
- ✅ Maintains public API compatibility
- ✅ Auto-generates sync version correctly
- ✅ Properly handles test updates

**Estimated effort**:
- Original estimate: 2-3 days
- With corrections: 2-3 days (same - test updates offset by easier sync generation)

**Risk level**: **Low-Medium**
- Low: Public API unchanged
- Medium: Internal structure changes require test updates

**Recommendation**: ✅ **Proceed with refactor after applying corrections**

The plan is well-thought-out and the corrections are straightforward. The key insight (use score_async directly) is correct and will significantly improve code maintainability.
