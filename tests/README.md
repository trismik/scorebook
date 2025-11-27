# Tests

## Structure

- `unit/` - Fast tests using mocks, no external dependencies
- `integration/` - Tests requiring a Trismik API key, network access or external services
- `extended/` - Longer-running tests for specific functionality

## Running Tests

```bash
# Unit tests (default)
pytest

# Integration tests
pytest tests/integration

# Extended tests
pytest tests/extended

# All tests
pytest tests/
```
