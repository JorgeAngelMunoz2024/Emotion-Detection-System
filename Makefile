.PHONY: help build-cpu build-gpu test test-audio test-personalized train-cpu train-gpu jupyter clean stop

help:
	@echo "Emotion Detection - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup:"
	@echo "  make build-cpu    - Build CPU Docker image"
	@echo "  make build-gpu    - Build GPU Docker image"
	@echo ""
	@echo "Testing:"
	@echo "  make test                - Run model verification tests"
	@echo "  make test-audio          - Run audio fusion tests"
	@echo "  make test-personalized   - Run personalized acoustic profiling tests"
	@echo ""
	@echo "Training:"
	@echo "  make train-cpu    - Start CPU training"
	@echo "  make train-gpu    - Start GPU training"
	@echo ""
	@echo "Development:"
	@echo "  make jupyter      - Start Jupyter notebook server"
	@echo ""
	@echo "Cleanup:"
	@echo "  make stop         - Stop all containers"
	@echo "  make clean        - Remove containers and images"
	@echo ""

build-cpu:
	@echo "Building CPU Docker image..."
	docker-compose build emotion-detector-cpu

build-gpu:
	@echo "Building GPU Docker image..."
	docker-compose build emotion-detector-gpu

test:
	@echo "Running model verification tests..."
	python test_models.py

test-audio:
	@echo "Running audio fusion logic tests..."
	docker-compose run --rm emotion-detector-cpu python3 tests/test_audio_fusion_logic.py

test-personalized:
	@echo "Running personalized acoustic profiling tests..."
	docker-compose run --rm emotion-detector-cpu python3 tests/test_personalized_acoustic_integration.py

train-cpu:
	@echo "Starting CPU training..."
	docker-compose up emotion-detector-cpu

train-gpu:
	@echo "Starting GPU training..."
	docker-compose up emotion-detector-gpu

jupyter:
	@echo "Starting Jupyter notebook server..."
	docker-compose up -d jupyter
	@echo ""
	@echo "Jupyter is running at http://localhost:8888"
	@echo "Getting access token..."
	@sleep 3
	@docker-compose logs jupyter | grep -A 5 "token" || echo "Check logs with: docker-compose logs jupyter"

stop:
	@echo "Stopping all containers..."
	docker-compose down

clean:
	@echo "Cleaning up containers and images..."
	docker-compose down --rmi all --volumes
	@echo "Removing checkpoints and logs..."
	rm -rf checkpoints/* logs/*
	@echo "Done!"

# Local development (without Docker)
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

test-local:
	@echo "Running tests locally..."
	python test_models.py

train-local:
	@echo "Starting training locally..."
	python train.py
