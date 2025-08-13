# Makefile for Krill project

.PHONY: test test-unit test-integration test-all clean

# Run all fast tests
test:
	pytest -m "not slow" -v

# Run unit tests only
test-unit:
	pytest -m unit -v

# Run integration tests only
test-integration:
	pytest -m integration -v

# Run all tests including slow ones
test-all:
	pytest -v

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete