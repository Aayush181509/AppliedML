# Makefile for Python and MkDocs commands
# Use bash for shell commands
SHELL := /bin/bash
# Default variables
INPUT_DIR  := notebooks


NOTEBOOK ?= notebooks/02_ml_workflow.ipynb
# Default output directory for converted notebooks
# If OUTPUT is not set, it defaults to docs/lectures/
OUTPUT   ?= docs/DCS404/
OUTPUT_DIR := docs/DCS404

# AI Module (Applied ML in Production)
AI_INPUT_DIR  := notebooks/ai-module
AI_OUTPUT_DIR := docs/AIModule
AI_NOTEBOOK   ?= notebooks/ai-module/00_setup_and_orientation.ipynb
AI_NOTEBOOKS  := $(wildcard $(AI_INPUT_DIR)/*.ipynb)

NOTEBOOKS  := $(wildcard $(INPUT_DIR)/*.ipynb)

.PHONY: run deploy build serve convert all convert-all convert-project convert-ai convert-ai-all

run:
	python main.py

deploy:
	mkdocs gh-deploy

build:
	mkdocs build

serve:
	mkdocs serve

convert:
	jupyter nbconvert --to markdown $(NOTEBOOK) --output-dir=$(OUTPUT)

# Batch convert all notebooks
convert-all:
	@echo "Converting all notebooks in $(INPUT_DIR)/ to Markdown..."
	@mkdir -p $(OUTPUT_DIR)
	@for nb in $(NOTEBOOKS); do \
		echo "Converting $$nb..."; \
		jupyter nbconvert --to markdown $$nb --output-dir=$(OUTPUT_DIR); \
	done

# Final project notebook (lives in notebooks/project/, not picked up by convert-all)
convert-project:
	jupyter nbconvert --to markdown notebooks/project/00_final_project.ipynb --output-dir=$(OUTPUT_DIR)/project

all: build serve

# Single AI-module notebook: make convert-ai AI_NOTEBOOK=notebooks/ai-module/01_problem_framing.ipynb
convert-ai:
	@mkdir -p $(AI_OUTPUT_DIR)
	jupyter nbconvert --to markdown $(AI_NOTEBOOK) --output-dir=$(AI_OUTPUT_DIR)

# All AI-module notebooks
convert-ai-all:
	@echo "Converting all notebooks in $(AI_INPUT_DIR)/ to Markdown..."
	@mkdir -p $(AI_OUTPUT_DIR)
	@for nb in $(AI_NOTEBOOKS); do \
		echo "Converting $$nb..."; \
		jupyter nbconvert --to markdown $$nb --output-dir=$(AI_OUTPUT_DIR); \
	done
