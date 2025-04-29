# Makefile for Player Performance Analysis

.PHONY: install run start_mlflow clean

# Variables
PYTHON = python3
MAIN = main.py
MLFLOW_PORT = 5000
MLFLOW_PID = mlflow.pid

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install torch torch_geometric pandas numpy opencv-python \
	scikit-learn matplotlib ultralytics mlflow mediapipe
	@echo "✅ Dependencies installed"

# Start MLflow server
start_mlflow:
	@echo "Starting MLflow server..."
	mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT) & echo $$! > $(MLFLOW_PID)
	@sleep 2
	@echo "✅ MLflow running at http://localhost:$(MLFLOW_PORT)"

# Run the project
run:
	@echo "Starting analysis..."
	$(PYTHON) $(MAIN)
	@echo "✅ Analysis completed"

# Clean
clean:
	@if [ -f "$(MLFLOW_PID)" ]; then \
		echo "Stopping MLflow server..."; \
		kill -9 `cat $(MLFLOW_PID)`; \
		rm $(MLFLOW_PID); \
		echo "✅ MLflow stopped"; \
	fi
	@echo "Cleaning up..."
	@rm -f mlflow.log
	@echo "✅ Cleanup completed"
