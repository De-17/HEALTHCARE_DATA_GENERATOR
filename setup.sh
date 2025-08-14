#!/bin/bash

# Synthetic Healthcare Data Generator Setup Script
# Automated setup for privacy-preserving medical data synthesis

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              🏥 SYNTHETIC HEALTHCARE DATA GENERATOR          ║"
    echo "║                                                              ║"
    echo "║          Privacy-Preserving AI for Medical Data Synthesis    ║"
    echo "║                                                              ║"
    echo "║  🔒 HIPAA/GDPR Compliant  |  🧠 Multiple AI Models          ║"
    echo "║  📊 15+ Evaluation Metrics |  🐳 Docker Ready               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        log_info "Python version: $PYTHON_VERSION"
        
        # Check if Python version is >= 3.8
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Python version is compatible (>= 3.8)"
        else
            log_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        log_success "pip3 found"
    else
        log_error "pip3 is not installed"
        exit 1
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        log_success "Docker found"
        DOCKER_AVAILABLE=true
    else
        log_warning "Docker not found - Docker features will be unavailable"
        DOCKER_AVAILABLE=false
    fi
    
    # Check Docker Compose (optional)
    if command -v docker-compose &> /dev/null; then
        log_success "Docker Compose found"
        DOCKER_COMPOSE_AVAILABLE=true
    else
        log_warning "Docker Compose not found - container orchestration unavailable"
        DOCKER_COMPOSE_AVAILABLE=false
    fi
    
    # Check system resources
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -ge 4 ]; then
            log_success "System memory: ${MEMORY_GB}GB (sufficient)"
        else
            log_warning "System memory: ${MEMORY_GB}GB (recommended: 8GB+)"
        fi
    fi
}

# Setup virtual environment
setup_environment() {
    log_step "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    log_info "Activating virtual environment..."
    source venv/bin/activate
    
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    log_success "Virtual environment ready"
}

# Install dependencies
install_dependencies() {
    log_step "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        pip install -r requirements.txt
        log_success "Dependencies installed successfully"
    else
        log_error "requirements.txt not found"
        exit 1
    fi
    
    # Install development dependencies if available
    if [ -f "requirements-dev.txt" ]; then
        read -p "Install development dependencies? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install -r requirements-dev.txt
            log_success "Development dependencies installed"
        fi
    fi
}

# Create directory structure
create_directories() {
    log_step "Creating project directory structure..."
    
    # Core directories
    mkdir -p data/{raw,processed,synthetic}
    mkdir -p models/{gans,vaes,saved}
    mkdir -p src/{data,models,evaluation,privacy,utils,visualization}
    mkdir -p notebooks
    mkdir -p tests
    mkdir -p configs
    mkdir -p results/{reports,figures,metrics}
    mkdir -p docker
    mkdir -p docs
    mkdir -p logs
    mkdir -p monitoring/{prometheus,grafana/provisioning/{dashboards,datasources}}
    mkdir -p nginx/{conf,ssl,logs}
    
    # Create __init__.py files
    find src -type d -exec touch {}/__init__.py \;
    
    # Create .gitkeep files for empty directories
    find . -type d -empty -exec touch {}/.gitkeep \;
    
    log_success "Directory structure created"
}

# Download sample datasets
download_sample_data() {
    log_step "Setting up sample datasets..."
    
    read -p "Download sample medical datasets? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Downloading UCI Heart Disease dataset..."
        
        # Create sample data download script
        cat > download_samples.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import requests
import os

def download_uci_heart_disease():
    """Download UCI Heart Disease dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        df = pd.read_csv(url, names=columns)
        df = df.replace('?', np.nan)
        df = df.dropna()
        
        # Convert target to binary
        df['target'] = (df['target'] > 0).astype(int)
        
        df.to_csv('data/raw/heart_disease.csv', index=False)
        print("✅ UCI Heart Disease dataset downloaded")
        return True
    except:
        print("❌ Failed to download UCI Heart Disease dataset")
        return False

def create_synthetic_medical_data():
    """Create synthetic medical dataset for demo"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic medical data
    ages = np.random.normal(55, 15, n_samples)
    ages = np.clip(ages, 18, 95).astype(int)
    
    data = {
        'age': ages,
        'gender': np.random.choice(['M', 'F'], n_samples),
        'bmi': np.random.normal(26.5, 4.2, n_samples),
        'blood_pressure': 90 + ages * 0.5 + np.random.normal(0, 15, n_samples),
        'cholesterol': 150 + ages * 1.2 + np.random.normal(0, 30, n_samples),
        'glucose': 80 + ages * 0.3 + np.random.normal(0, 20, n_samples),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'exercise': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/synthetic_medical_demo.csv', index=False)
    print("✅ Synthetic medical demo dataset created")

def create_breast_cancer_data():
    """Create breast cancer dataset from sklearn"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    
    df.to_csv('data/raw/breast_cancer.csv', index=False)
    print("✅ Breast cancer dataset created")

if __name__ == "__main__":
    print("📊 Downloading sample datasets...")
    os.makedirs('data/raw', exist_ok=True)
    
    download_uci_heart_disease()
    create_synthetic_medical_data()
    create_breast_cancer_data()
    
    print("✅ All sample datasets ready!")
EOF
        
        python download_samples.py
        rm download_samples.py
        
        log_success "Sample datasets downloaded"
    else
        log_info "Skipping sample dataset download"
    fi
}

# Setup configuration files
setup_configs() {
    log_step "Setting up configuration files..."
    
    # Create .env file
    cat > .env << 'EOF'
# Synthetic Healthcare Data Generator Configuration

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=True

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MAX_UPLOAD_SIZE=100MB
CORS_ORIGINS=*

# Database Configuration (PostgreSQL)
DATABASE_URL=postgresql://synthetic_user:synthetic_secure_password@localhost:5432/synthetic_healthcare
POSTGRES_USER=synthetic_user
POSTGRES_PASSWORD=synthetic_secure_password
POSTGRES_DB=synthetic_healthcare

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production
JWT_EXPIRY_HOURS=24

# Privacy Settings
DEFAULT_PRIVACY_LEVEL=medium
DEFAULT_COMPLIANCE_MODE=hipaa
ENABLE_AUDIT_LOGGING=true

# Model Training
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=64
MAX_TRAINING_TIME_HOURS=6

# Monitoring
ENABLE_PROMETHEUS_METRICS=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EOF

    log_info "Configuration file (.env) created"
    
    # Create basic prometheus config
    mkdir -p monitoring
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'synthetic-healthcare'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

    log_success "Configuration files created"
}

# Setup Docker environment
setup_docker() {
    if [ "$DOCKER_AVAILABLE" = true ]; then
        log_step "Setting up Docker environment..."
        
        read -p "Build Docker images? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Building Docker images..."
            docker build -t synthetic-healthcare:latest .
            log_success "Docker image built"
            
            if [ "$DOCKER_COMPOSE_AVAILABLE" = true ]; then
                log_info "Setting up Docker Compose..."
                docker-compose --profile production build
                log_success "Docker Compose images built"
            fi
        fi
    else
        log_info "Docker not available, skipping Docker setup"
    fi
}

# Run initial tests
run_tests() {
    log_step "Running initial tests..."
    
    read -p "Run basic functionality tests? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Running demo script..."
        
        if python run_demo.py; then
            log_success "Demo completed successfully!"
        else
            log_error "Demo failed. Check the logs for details."
        fi
    fi
}

# Display setup completion info
show_completion_info() {
    echo -e "\n${GREEN}🎉 SETUP COMPLETED SUCCESSFULLY!${NC}\n"
    
    echo -e "${BLUE}📋 Next Steps:${NC}"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Start the API server: python api_server.py"
    echo "3. Open browser: http://localhost:8000/docs"
    echo "4. Run demo: python run_demo.py"
    
    echo -e "\n${BLUE}🐳 Docker Commands:${NC}"
    echo "• Development: docker-compose --profile dev up -d"
    echo "• Production: docker-compose --profile production up -d"
    echo "• With monitoring: docker-compose --profile production --profile monitoring up -d"
    
    echo -e "\n${BLUE}📊 Available Endpoints:${NC}"
    echo "• API Documentation: http://localhost:8000/docs"
    echo "• Health Check: http://localhost:8000/health"
    echo "• Grafana Dashboard: http://localhost:3000 (admin/synthetic_admin_2024)"
    echo "• Prometheus: http://localhost:9090"
    
    echo -e "\n${BLUE}📁 Important Files:${NC}"
    echo "• Configuration: .env"
    echo "• Sample data: data/raw/"
    echo "• API server: api_server.py"
    echo "• Demo script: run_demo.py"
    
    echo -e "\n${BLUE}🔒 Security Notes:${NC}"
    echo "• Change default passwords in .env before production"
    echo "• Review privacy settings for your use case"
    echo "• Enable HTTPS in production environments"
    
    echo -e "\n${PURPLE}💡 Pro Tips:${NC}"
    echo "• Use 'high' privacy level for sensitive medical data"
    echo "• Monitor GPU usage for training acceleration"
    echo "• Scale with Docker Compose for production workloads"
    
    echo -e "\n${GREEN}Happy generating synthetic healthcare data! 🏥✨${NC}"
}

# Main setup function
main() {
    print_banner
    
    log_info "Starting Synthetic Healthcare Data Generator setup..."
    log_info "Setup started at: $(date)"
    
    # Run setup steps
    check_requirements
    setup_environment
    install_dependencies
    create_directories
    setup_configs
    download_sample_data
    setup_docker
    run_tests
    
    show_completion_info
    
    log_success "Setup completed at: $(date)"
}

# Handle script interruption
trap 'echo -e "\n${RED}Setup interrupted by user${NC}"; exit 1' INT

# Run main setup
main "$@"