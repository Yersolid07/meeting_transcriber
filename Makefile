# =============================================================================
# Makefile - Meeting Transcriber
# =============================================================================

.PHONY: install test clean run evaluate docs help

# Variables
PYTHON = python
PIP = pip
VENV = venv
SRC_DIR = src
TEST_DIR = tests
DATA_DIR = data
OUTPUT_DIR = data/output

# Default target
help:
	@echo "Meeting Transcriber - Available Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install dev dependencies"
	@echo "  make setup-dirs   - Create directory structure"
	@echo ""
	@echo "Running:"
	@echo "  make run AUDIO=path/to/audio.wav     - Run transcription"
	@echo "  make run-batch DIR=path/to/folder    - Batch processing"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate     - Run full evaluation"
	@echo "  make create-gt    - Create ground truth templates"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run unit tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean generated files"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Format code"
	@echo ""

# Setup
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 isort

setup-dirs:
	mkdir -p $(DATA_DIR)/audio/kondisi_bersih
	mkdir -p $(DATA_DIR)/audio/kondisi_noisy
	mkdir -p $(DATA_DIR)/audio/kondisi_overlap
	mkdir -p $(DATA_DIR)/audio/kondisi_multispeaker
	mkdir -p $(DATA_DIR)/ground_truth/kondisi_bersih
	mkdir -p $(DATA_DIR)/ground_truth/kondisi_noisy
	mkdir -p $(DATA_DIR)/ground_truth/kondisi_overlap
	mkdir -p $(DATA_DIR)/ground_truth/kondisi_multispeaker
	mkdir -p $(DATA_DIR)/output
	mkdir -p models
	mkdir -p cache
	mkdir -p logs
	@echo "✓ Directory structure created"

# Running
run:
ifndef AUDIO
	@echo "Error: AUDIO not specified"
	@echo "Usage: make run AUDIO=path/to/audio.wav"
else
	$(PYTHON) main.py --audio $(AUDIO) --title "Meeting" $(ARGS)
endif

run-batch:
ifndef DIR
	@echo "Error: DIR not specified"
	@echo "Usage: make run-batch DIR=path/to/folder"
else
	$(PYTHON) main.py --batch $(DIR) --output $(OUTPUT_DIR) $(ARGS)
endif

# Evaluation
evaluate:
	$(PYTHON) scripts/run_evaluation.py --output $(OUTPUT_DIR)/evaluation

create-gt:
ifndef AUDIO
	@echo "Error: AUDIO not specified"
	@echo "Usage: make create-gt AUDIO=path/to/audio.wav OUTPUT=path/to/output"
else
	$(PYTHON) scripts/create_gt_template.py --audio $(AUDIO) --output $(OUTPUT)
endif

# Testing
test:
	$(PYTHON) -m pytest $(TEST_DIR) -v

test-cov:
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html

# Utilities
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "✓ Cleaned"

clean-output:
	rm -rf $(OUTPUT_DIR)/*
	@echo "✓ Output cleaned"

lint:
	flake8 $(SRC_DIR) --max-line-length=100

format:
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

# Documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here
