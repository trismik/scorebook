# Unasync Implementation Plan for Scorebook

## Overview

This document outlines the implementation plan for integrating `unasync` into the Scorebook project to automatically generate synchronous code from asynchronous implementations, eliminating code duplication and ensuring API parity.

## Current State Analysis

### ✅ What's Already Done
- Comprehensive async implementation in `src/scorebook/evaluate/_async/evaluate_async.py`
- Shared helper functions in `src/scorebook/evaluate/evaluate_helpers.py`
- Directory structure with `_sync` and `_async` folders
- Both `evaluate` and `evaluate_async` exposed in main API
- Empty `run_unasync.py` file in root directory

### 🔄 What Needs Implementation
- Unasync dependency and configuration
- Auto-generation of sync code from async
- API reorganization and deprecation
- Build process integration

## Implementation Strategy

### Phase 1: Unasync Infrastructure Setup

#### Task 1: Add Dependencies
- Add `unasync` to dev dependencies in `pyproject.toml`
- Install and verify unasync functionality

#### Task 2: Configure Transformation Rules
Configure `run_unasync.py` with the following transformation rules:

```python
import unasync

# Key transformation rules needed:
rules = [
    # Function/method transformations
    unasync.Rule("async def", "def"),
    unasync.Rule("await ", ""),

    # Import/module transformations
    unasync.Rule("_async", "_sync"),
    unasync.Rule("evaluate_async", "evaluate"),
    unasync.Rule("execute_runs_async", "execute_runs"),
    unasync.Rule("execute_run_async", "execute_run"),
    unasync.Rule("execute_classic_eval_run_async", "execute_classic_eval_run"),
    unasync.Rule("run_inference_callable_async", "run_inference_callable"),
    unasync.Rule("execute_adaptive_eval_run_async", "execute_adaptive_eval_run"),
    unasync.Rule("upload_classic_run_async", "upload_classic_run"),

    # Concurrency transformations
    unasync.Rule("asyncio.gather", "map"),
    unasync.Rule("import asyncio", "from concurrent.futures import ThreadPoolExecutor"),
]
```

#### Task 3: Generate Sync Code
- Create `src/scorebook/evaluate/_sync/evaluate.py` from async source
- Verify generated code is syntactically correct
- Test basic functionality

### Phase 2: API Reorganization

#### Task 4: Update Import Structure
Update `src/scorebook/evaluate/__init__.py`:
```python
# Import from generated sync module
from ._sync.evaluate import evaluate
# Import from async module
from ._async.evaluate_async import evaluate_async

__all__ = ["evaluate", "evaluate_async"]
```

#### Task 5: Update Main Package Imports
Update `src/scorebook/__init__.py`:
```python
# Route through new evaluate module structure
from scorebook.evaluate import evaluate, evaluate_async
```

#### Task 6: Add Deprecation Warning
Add deprecation warning to current `src/scorebook/evaluate.py`:
```python
import warnings
warnings.warn(
    "Direct import from scorebook.evaluate is deprecated. "
    "Use 'from scorebook import evaluate' instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### Phase 3: Build Integration

#### Task 7: Add to Build Process
Update `pyproject.toml` to run unasync during build:
```toml
[build-system]
requires = ["poetry-core", "unasync"]
build-backend = "poetry.core.masonry.api"

[tool.poetry.scripts]
generate-sync = "run_unasync:main"
```

#### Task 8: Update Pre-commit Hooks
Add unasync generation to `.pre-commit-hooks.yaml`:
```yaml
- id: generate-sync-code
  name: Generate sync code from async
  entry: python run_unasync.py
  language: python
  files: src/scorebook/evaluate/_async/.*\.py$
```

#### Task 9: Update CI/CD Pipeline
Ensure sync code generation runs before tests:
```yaml
steps:
  - name: Generate sync code
    run: python run_unasync.py
  - name: Run tests
    run: pytest
```

### Phase 4: Testing and Validation

#### Task 10: API Parity Tests
Create tests to ensure sync and async APIs behave identically:
- Same function signatures
- Same return values
- Same error handling
- Same performance characteristics (where applicable)

#### Task 11: Update Examples
Update all examples to use the new API structure:
- Verify both sync and async examples work
- Update documentation and docstrings
- Test with actual model inference

#### Task 12: Integration Testing
- Test with existing Scorebook consumers
- Verify backward compatibility
- Performance benchmarking

### Phase 5: Cleanup

#### Task 13: Remove Old Code
- Remove deprecated `src/scorebook/evaluate.py`
- Clean up unused imports
- Update documentation

#### Task 14: Documentation Updates
- Update API documentation
- Add unasync development notes
- Update contribution guidelines

## File Structure After Implementation

```
src/scorebook/
├── __init__.py                          # Main package exports
├── evaluate/
│   ├── __init__.py                      # Exports both sync & async
│   ├── _async/
│   │   ├── __init__.py
│   │   └── evaluate_async.py           # Source of truth (async)
│   ├── _sync/
│   │   ├── __init__.py
│   │   └── evaluate.py                 # Auto-generated (sync)
│   └── evaluate_helpers.py             # Shared utilities
├── eval_dataset.py
├── inference_pipeline.py
└── [other modules...]

run_unasync.py                           # Unasync configuration
```

## Key Benefits

1. **Single Source of Truth**: Only async code needs to be maintained
2. **Automatic Sync Generation**: No manual sync code duplication
3. **API Compatibility**: Both `evaluate()` and `evaluate_async()` available
4. **Easy Maintenance**: Changes only needed in async code
5. **Type Safety**: Generated code inherits type annotations
6. **Performance**: Sync code optimized for synchronous execution

## Development Workflow

1. **Make changes only to async code** in `_async/evaluate_async.py`
2. **Run unasync** to regenerate sync code: `python run_unasync.py`
3. **Test both APIs** to ensure parity
4. **Commit both async source and generated sync code**

## Risk Mitigation

- **Generated code in version control**: Ensures builds work without unasync
- **Comprehensive tests**: Verify sync/async parity
- **Gradual migration**: Deprecation warnings before removal
- **Backward compatibility**: Existing APIs continue to work

## Success Criteria

- [ ] Sync code successfully generated from async source
- [ ] All tests pass for both sync and async APIs
- [ ] No breaking changes for existing users
- [ ] Documentation updated and complete
- [ ] CI/CD pipeline includes unasync generation
- [ ] Performance benchmarks show no regression
