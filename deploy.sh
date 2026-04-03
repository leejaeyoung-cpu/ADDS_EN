#!/bin/bash
# ADDS Deployment Script for Linux/Mac

set -e  # Exit on error

echo "🚀 ADDS Deployment Script"
echo "=========================="

# Configuration
PROJECT_NAME="adds"
ENV_FILE=".env"
DOCKER_COMPOSE_FILE="docker-compose.yml"
GPU_COMPOSE_FILE="docker-compose.gpu.yml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
function print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Docker
print_step "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker version: $(docker --version)"
echo "✓ Docker Compose version: $(docker-compose --version)"

# Check environment file
print_step "Checking environment configuration..."
if [ ! -f "$ENV_FILE" ]; then
    print_warning "No .env file found. Copying from .env.example..."
    cp .env.example .env
    print_warning "⚠️  Please edit .env file with your configuration before continuing."
    read -p "Press Enter to continue after editing .env..."
fi

# GPU detection
print_step "Detecting GPU..."
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        echo "✓ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
fi

# Choose deployment mode
if [ "$GPU_AVAILABLE" = true ]; then
    read -p "Deploy with GPU support? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DOCKER_COMPOSE_FILE="$GPU_COMPOSE_FILE"
        print_step "Using GPU-enabled configuration"
    fi
fi

# Build images
print_step "Building Docker images..."
docker-compose -f $DOCKER_COMPOSE_FILE build --no-cache

if [ $? -ne 0 ]; then
    print_error "Docker build failed!"
    exit 1
fi

# Stop existing containers
print_step "Stopping existing containers..."
docker-compose -f $DOCKER_COMPOSE_FILE down

# Start services
print_step "Starting ADDS services..."
docker-compose -f $DOCKER_COMPOSE_FILE up -d

if [ $? -ne 0 ]; then
    print_error "Failed to start services!"
    exit 1
fi

# Wait for services to be healthy
print_step "Waiting for services to be healthy..."
sleep 10

# Health checks
print_step "Running health checks..."

echo "Checking API..."
API_HEALTH=$(curl -s http://localhost:8000/health || echo "failed")
if [[ $API_HEALTH == *"healthy"* ]]; then
    echo "✓ API is healthy"
else
    print_error "API health check failed!"
    docker-compose -f $DOCKER_COMPOSE_FILE logs adds-api
    exit 1
fi

echo "Checking UI..."
if curl -s -f http://localhost:8501/_stcore/health > /dev/null; then
    echo "✓ UI is healthy"
else
    print_warning "UI health check could not be verified (might still be starting)"
fi

# Show logs
print_step "Deployment complete! 🎉"
echo ""
echo "Services are running:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/api/docs"
echo "  - UI: http://localhost:8501"
echo ""
echo "To view logs:"
echo "  docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose -f $DOCKER_COMPOSE_FILE down"
echo ""

# Optional: Show container status
docker-compose -f $DOCKER_COMPOSE_FILE ps
