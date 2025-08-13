# Krill Test Suite

This directory contains comprehensive tests for the Krill project's resume functionality.

## Test Structure

### Unit Tests (`test_resume_unit.py`)
- Tests individual functions in isolation
- Uses mocks to simulate dependencies
- Fast execution (~5 seconds)
- Tests all resume options and edge cases

### Integration Tests (`test_resume_integration.py`)
- Tests actual file parsing and real data processing
- Tests the integration between components
- Moderate execution time
- Verifies real parsing logic works correctly
- Includes end-to-end tests with real remote repositories

## Running Tests

### Run All Fast Tests (Unit + Integration)
```bash
pytest tests/ -m "not slow"
```

### Run Unit Tests Only
```bash
pytest tests/test_resume_unit.py
```

### Run Integration Tests Only
```bash
pytest tests/test_resume_integration.py
```

### Run All Tests
```bash
pytest tests/
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_resume_unit.py::TestResumeFunctionality::test_determine_resume_checkpoint_false
```

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow tests

## Test Coverage

The test suite covers:

1. **All Resume Options**:
   - `false` - Start from scratch
   - `true` - Let Hugging Face auto-detect local checkpoint
   - `auto` - Smart detection (local vs remote)
   - `local` - Resume from local checkpoint only
   - `remote` - Resume from remote checkpoint only

2. **All Checkpoint States**:
   - Local checkpoint exists/not exists
   - Remote checkpoint exists/not exists
   - Remote checkpoint with/without step information
   - Local newer than remote and vice versa
   - Only local exists, only remote exists, neither exists

3. **Error Handling**:
   - Invalid resume options
   - Missing checkpoints
   - Network errors
   - File system errors

4. **End-to-End Scenarios**:
   - Real remote repository testing with `pretraining/krill-e2e-ci-pico`
   - Verification that remote checkpoint step is correctly identified as 44
   - Complete workflow testing from checkpoint detection to resume path determination