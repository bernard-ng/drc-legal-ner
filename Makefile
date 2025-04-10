.PHONY: default
default: help

.PHONY: help
help:
	@echo Tasks:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: train_efficiency
train_efficiency: ## Train the model with efficiency
	python -m spacy train config_efficiency.cfg --output ./models/efficiency --paths.train ./dataset/spacy/train.spacy --paths.dev ./dataset/spacy/dev.spacy

.PHONY: train_accuracy
train_accuracy: ## Train the model with accuracy
	python -m spacy train config_accuracy.cfg --output ./models/accuracy --paths.train ./dataset/spacy/train.spacy --paths.dev ./dataset/spacy/dev.spacy

.PHONY: evaluate
evaluate: ## Evaluate the model
	python -m spacy evaluate ./models/model-best ./dataset/spacy/dev.spacy --output ./results/evaluation.json

.PHONY: benchmark
benchmark: ## Benchmark the model
	python -m spacy benchmark accuracy ./models/model-best ./dataset/spacy/dev.spacy --output ./results/benchmark.json

.PHONY: visualize
visualize: ## Visualize NER
	streamlit run app.py

.PHONY: clean
clean: ## Clean the model and results
	rm -rf ./models
	rm -rf ./results
	rm -rf ./dataset/spacy/train.spacy
	rm -rf ./dataset/spacy/dev.spacy
