# Project Structure

## Main Files
- `main.py` - Main FastAPI application entry point
- `Dockerfile` - Docker image configuration for Kubernetes
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Application Structure
- `api/` - FastAPI endpoint definitions
- `config/` - Configuration and settings
- `core/` - Core application initialization
- `models/` - Data models and schemas
- `services/` - Business logic services
- `utils/` - Utility functions

## Kubernetes Configuration
- `k8s/base/` - Base Kubernetes manifests
- `k8s/overlays/minikube/` - Minikube-specific patches

## Deployment
- `deploy-minikube.bat` - Windows deployment script
- `cleanup-minikube.bat` - Windows cleanup script

## Model Weights
- `synt_ticket_model_weights.pth` - Pre-trained model weights

## Testing
- `test_duckduckgo.py` - DuckDuckGo search service tests
