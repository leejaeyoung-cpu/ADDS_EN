# ADDS Deployment Script for Windows PowerShell
# Run with: .\deploy.ps1

param(
    [switch]$GPU = $false,
    [switch]$NoBuild = $false
)

Write-Host "🚀 ADDS Deployment Script" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green
Write-Host ""

# Configuration
$PROJECT_NAME = "adds"
$ENV_FILE = ".env"
$DOCKER_COMPOSE_FILE = "docker-compose.yml"
$GPU_COMPOSE_FILE = "docker-compose.gpu.yml"

# Functions
function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check Docker
Write-Step "Checking Docker installation..."
try {
    $dockerVersion = docker --version
    $composeVersion = docker-compose --version
    Write-Success "Docker version: $dockerVersion"
    Write-Success "Docker Compose version: $composeVersion"
} catch {
    Write-ErrorMsg "Docker or Docker Compose is not installed!"
    Write-Host "Please install Docker Desktop for Windows from: https://www.docker.com/products/docker-desktop"
    exit 1
}

# Check if Docker is running
try {
    docker ps | Out-Null
    Write-Success "Docker daemon is running"
} catch {
    Write-ErrorMsg "Docker daemon is not running! Please start Docker Desktop."
    exit 1
}

# Check environment file
Write-Step "Checking environment configuration..."
if (-not (Test-Path $ENV_FILE)) {
    Write-Warning "No .env file found. Copying from .env.example..."
    Copy-Item ".env.example" $ENV_FILE
    Write-Warning "⚠️  Please edit .env file with your configuration."
    $continue = Read-Host "Press Enter to continue after editing .env, or 'q' to quit"
    if ($continue -eq 'q') { exit 0 }
}

# GPU detection
Write-Step "Detecting GPU..."
$GPUAvailable = $false
try {
    $nvidiaSmi = nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        $GPUAvailable = $true
        Write-Success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    }
} catch {
    Write-Host "No NVIDIA GPU detected (CPU mode will be used)"
}

# Choose deployment mode
if ($GPUAvailable -and -not $GPU) {
    $useGPU = Read-Host "Deploy with GPU support? (y/n)"
    if ($useGPU -eq 'y' -or $useGPU -eq 'Y') {
        $GPU = $true
    }
}

if ($GPU) {
    $DOCKER_COMPOSE_FILE = $GPU_COMPOSE_FILE
    Write-Step "Using GPU-enabled configuration"
    
    # Check NVIDIA Container Runtime
    try {
        docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
        Write-Success "NVIDIA Container Runtime is properly configured"
    } catch {
        Write-Warning "NVIDIA Container Runtime might not be configured correctly"
    }
}

# Build images
if (-not $NoBuild) {
    Write-Step "Building Docker images..."
    docker-compose -f $DOCKER_COMPOSE_FILE build --no-cache
    
    if ($LASTEXITCODE -ne 0) {
        Write-ErrorMsg "Docker build failed!"
        exit 1
    }
    Write-Success "Build complete"
}

# Stop existing containers
Write-Step "Stopping existing containers..."
docker-compose -f $DOCKER_COMPOSE_FILE down

# Start services
Write-Step "Starting ADDS services..."
docker-compose -f $DOCKER_COMPOSE_FILE up -d

if ($LASTEXITCODE -ne 0) {
    Write-ErrorMsg "Failed to start services!"
    exit 1
}

# Wait for services
Write-Step "Waiting for services to start..."
Start-Sleep -Seconds 15

# Health checks
Write-Step "Running health checks..."

Write-Host "Checking API..." -NoNewline
try {
    $apiHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    if ($apiHealth.status -eq "healthy") {
        Write-Success " API is healthy"
    } else {
        Write-ErrorMsg " API health check failed!"
        docker-compose -f $DOCKER_COMPOSE_FILE logs adds-api
exit 1
    }
} catch {
    Write-ErrorMsg " API is not responding!"
    docker-compose -f $DOCKER_COMPOSE_FILE logs adds-api
    exit 1
}

Write-Host "Checking UI..." -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Success " UI is healthy"
    }
} catch {
    Write-Warning " UI health check could not be verified (might still be starting)"
}

# Deployment complete
Write-Host ""
Write-Host "═══════════════════════════════════════" -ForegroundColor Green
Write-Host "  Deployment Complete! 🎉" -ForegroundColor Green
Write-Host "═══════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Services are running:" -ForegroundColor Cyan
Write-Host "  - API: http://localhost:8000" -ForegroundColor White
Write-Host "  - API Docs: http://localhost:8000/api/docs" -ForegroundColor White
Write-Host "  - UI: http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  View logs:  docker-compose -f $DOCKER_COMPOSE_FILE logs -f" -ForegroundColor White
Write-Host "  Stop:       docker-compose -f $DOCKER_COMPOSE_FILE down" -ForegroundColor White
Write-Host "  Restart:    docker-compose -f $DOCKER_COMPOSE_FILE restart" -ForegroundColor White
Write-Host ""

# Show container status
docker-compose -f $DOCKER_COMPOSE_FILE ps
