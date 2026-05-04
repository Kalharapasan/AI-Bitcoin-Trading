#!/usr/bin/env python3
"""
Setup script for Bitcoin Trading AI application.
Handles installation, dependencies, and initial setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import yaml
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class SetupManager:
    """Manager for setting up the Bitcoin Trading AI application"""
    
    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.requirements_file = self.project_root / "requirements.txt"
        
        # Create logger
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logging for the setup process"""
        import logging
        logging.basicConfig(
            level=logging.DEBUG if self.args.verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        self.logger.info("Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("Python 3.8 or higher is required")
            return False
        
        # Check if pip is available
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         capture_output=True, check=True)
        except subprocess.CalledProcessError:
            self.logger.error("pip is not available")
            return False
        
        self.logger.info("✓ Prerequisites check passed")
        return True
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        self.logger.info("Creating project directories...")
        
        directories = [
            "data/raw",
            "data/processed", 
            "data/cache",
            "models",
            "logs",
            "results/backtesting",
            "results/live_trading",
            "results/analysis",
            "backups",
            "config",
            "scripts",
            "tests/test_data"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {full_path}")
        
        self.logger.info("✓ Directories created")
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        self.logger.info("Installing dependencies...")
        
        if not self.requirements_file.exists():
            self.logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            # Upgrade pip first
            self.logger.info("Upgrading pip...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=not self.args.verbose
            )
            
            # Install requirements
            self.logger.info("Installing from requirements.txt...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                check=True,
                capture_output=not self.args.verbose
            )
            
            # Install development dependencies if in dev mode
            if self.args.dev:
                self.logger.info("Installing development dependencies...")
                dev_packages = [
                    "pytest",
                    "pytest-cov",
                    "black",
                    "flake8",
                    "mypy",
                    "pre-commit"
                ]
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + dev_packages,
                    check=True,
                    capture_output=not self.args.verbose
                )
            
            self.logger.info("✓ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            if e.stderr:
                self.logger.error(f"Error details: {e.stderr.decode()}")
            return False
    
    def create_configuration_files(self) -> bool:
        """Create configuration files from templates"""
        self.logger.info("Creating configuration files...")
        
        # Create .env file from .env.example
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"
        
        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            self.logger.info(f"Created {env_file} from template")
        
        # Create config.yaml if it doesn't exist
        config_yaml = self.config_dir / "config.yaml"
        if not config_yaml.exists():
            default_config = self.generate_default_config()
            with open(config_yaml, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            self.logger.info(f"Created default configuration: {config_yaml}")
        
        # Create config/__init__.py
        init_file = self.config_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()
        
        # Create database config
        db_config = self.config_dir / "database_config.json"
        if not db_config.exists():
            default_db_config = {
                "sqlite": {
                    "database": "trading_ai.db",
                    "echo": False
                },
                "postgresql": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "trading_ai",
                    "username": "postgres",
                    "password": "",
                    "pool_size": 10
                }
            }
            with open(db_config, 'w') as f:
                json.dump(default_db_config, f, indent=2)
        
        self.logger.info("✓ Configuration files created")
        return True
    
    def generate_default_config(self) -> Dict[str, Any]:
        """Generate default configuration"""
        return {
            'app': {
                'name': 'Bitcoin Trading AI',
                'version': '1.0.0',
                'environment': 'development',
                'debug': True,
                'log_level': 'INFO'
            },
            'database': {
                'type': 'sqlite',
                'database': 'trading_ai.db',
                'host': 'localhost',
                'port': 5432,
                'username': 'postgres',
                'password': '',
                'pool_size': 10,
                'echo': False
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'password': '',
                'db': 0
            },
            'exchange': {
                'name': 'binance',
                'api_key': 'YOUR_API_KEY',
                'api_secret': 'YOUR_API_SECRET',
                'testnet': True,
                'rate_limit': True
            },
            'trading': {
                'symbol': 'BTC/USDT',
                'timeframe': '1h',
                'initial_capital': 10000.0,
                'risk_per_trade': 0.02,
                'max_position_size': 0.1,
                'commission_rate': 0.001
            },
            'models': {
                'transformer': {
                    'd_model': 64,
                    'nhead': 4,
                    'num_layers': 3,
                    'dropout': 0.1
                },
                'lstm': {
                    'hidden_size': 50,
                    'num_layers': 2,
                    'dropout': 0.2
                }
            },
            'backtesting': {
                'initial_capital': 10000.0,
                'commission': 0.001,
                'slippage': 0.0001,
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            },
            'monitoring': {
                'metrics_port': 9090,
                'alert_email': 'alerts@example.com',
                'check_interval': 60
            },
            'web': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': True,
                'secret_key': 'change-this-in-production'
            }
        }
    
    def initialize_database(self) -> bool:
        """Initialize database schema"""
        self.logger.info("Initializing database...")
        
        try:
            # Import database modules
            from database.connection import init_database
            from config.config_manager import ConfigManager
            
            # Initialize configuration
            config_manager = ConfigManager()
            
            # Initialize database
            if init_database(config_manager, drop_existing=self.args.force):
                self.logger.info("✓ Database initialized successfully")
                return True
            else:
                self.logger.error("Failed to initialize database")
                return False
                
        except ImportError as e:
            self.logger.error(f"Failed to import database modules: {e}")
            self.logger.info("Make sure dependencies are installed first (use --install-deps)")
            return False
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False
    
    def setup_git_hooks(self) -> None:
        """Setup git hooks for development"""
        if not self.args.dev:
            return
            
        self.logger.info("Setting up git hooks...")
        
        # Create pre-commit hook
        git_hooks_dir = self.project_root / ".git" / "hooks"
        if git_hooks_dir.exists():
            pre_commit_hook = git_hooks_dir / "pre-commit"
            pre_commit_content = """#!/bin/bash
# Bitcoin Trading AI pre-commit hook

echo "Running pre-commit checks..."

# Run black formatting check
echo "Checking code formatting..."
python -m black --check --diff .

if [ $? -ne 0 ]; then
    echo "Code formatting issues found. Run 'black .' to fix."
    exit 1
fi

# Run flake8 linting
echo "Running linting..."
python -m flake8 .

if [ $? -ne 0 ]; then
    echo "Linting issues found."
    exit 1
fi

# Run tests
echo "Running tests..."
python -m pytest tests/ -xvs

if [ $? -ne 0 ]; then
    echo "Tests failed."
    exit 1
fi

echo "All checks passed!"
exit 0
"""
            
            with open(pre_commit_hook, 'w') as f:
                f.write(pre_commit_content)
            
            # Make executable
            pre_commit_hook.chmod(0o755)
            self.logger.info("✓ Git pre-commit hook installed")
    
    def run_tests(self) -> bool:
        """Run test suite"""
        self.logger.info("Running tests...")
        
        try:
            import pytest
            
            test_dir = self.project_root / "tests"
            if not test_dir.exists():
                self.logger.warning("Test directory not found, skipping tests")
                return True
            
            # Run tests
            test_args = [
                str(test_dir),
                '-v',
                '--tb=short'
            ]
            
            if self.args.verbose:
                test_args.append('-s')
            
            exit_code = pytest.main(test_args)
            
            if exit_code == 0:
                self.logger.info("✓ All tests passed")
                return True
            else:
                self.logger.error(f"Tests failed with exit code: {exit_code}")
                return False
                
        except ImportError:
            self.logger.warning("pytest not installed, skipping tests")
            return True
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return False
    
    def create_docker_files(self) -> None:
        """Create Docker configuration files"""
        self.logger.info("Creating Docker configuration...")
        
        # Dockerfile
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/cache models logs results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 9090

# Run the application
CMD ["python", "main.py"]
"""
        
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            with open(dockerfile, 'w') as f:
                f.write(dockerfile_content)
        
        # docker-compose.yml
        compose_content = """version: '3.8'

services:
  trading-app:
    build: .
    ports:
      - "8000:8000"  # Web dashboard
      - "9090:9090"  # Metrics
    environment:
      - DATABASE_TYPE=postgresql
      - REDIS_HOST=redis
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./results:/app/results
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - trading-network

  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=trading_ai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - trading-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - trading-app
    networks:
      - trading-network

volumes:
  postgres-data:
  redis-data:
  grafana-data:

networks:
  trading-network:
    driver: bridge
"""
        
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.exists():
            with open(compose_file, 'w') as f:
                f.write(compose_content)
        
        self.logger.info("✓ Docker configuration created")
    
    def create_utility_scripts(self) -> None:
        """Create utility scripts"""
        self.logger.info("Creating utility scripts...")
        
        scripts_dir = self.project_root / "scripts"
        
        # create_venv.sh
        venv_script = scripts_dir / "create_venv.sh"
        venv_content = """#!/bin/bash
# Create virtual environment for Bitcoin Trading AI

echo "Creating virtual environment..."

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null; then
    echo "virtualenv not found. Installing..."
    pip install virtualenv
fi

# Create virtual environment
virtualenv venv -p python3

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "Virtual environment created successfully!"
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"
"""
        
        if not venv_script.exists():
            with open(venv_script, 'w') as f:
                f.write(venv_content)
            venv_script.chmod(0o755)
        
        # backup_database.py
        backup_script = scripts_dir / "backup_database.py"
        backup_content = '''#!/usr/bin/env python3
"""
Database backup script for Bitcoin Trading AI.
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import get_database_manager
from config.config_manager import ConfigManager

def backup_database():
    """Create a backup of the database"""
    print("Starting database backup...")
    
    try:
        # Initialize configuration
        config = ConfigManager()
        
        # Get database manager
        db_manager = get_database_manager(config)
        
        # Create backup directory
        backup_dir = project_root / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Create timestamp for backup file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup database
        backup_path = db_manager.backup_database(
            backup_path=str(backup_dir / f"trading_ai_backup_{timestamp}.db")
        )
        
        if backup_path:
            print(f"✓ Database backed up to: {backup_path}")
            
            # Keep only last 7 backups
            backups = sorted(backup_dir.glob("trading_ai_backup_*.db"))
            if len(backups) > 7:
                for old_backup in backups[:-7]:
                    old_backup.unlink()
                    print(f"Removed old backup: {old_backup.name}")
                    
            return True
        else:
            print("✗ Database backup failed")
            return False
            
    except Exception as e:
        print(f"✗ Backup error: {e}")
        return False

if __name__ == "__main__":
    success = backup_database()
    sys.exit(0 if success else 1)
'''
        
        if not backup_script.exists():
            with open(backup_script, 'w') as f:
                f.write(backup_content)
            backup_script.chmod(0o755)
        
        self.logger.info("✓ Utility scripts created")
    
    def print_setup_summary(self) -> None:
        """Print setup summary"""
        print("\n" + "="*60)
        print("BITCOIN TRADING AI SETUP COMPLETE!")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Configure your environment:")
        print(f"   Edit {self.project_root / '.env'} with your API keys")
        print(f"   Review {self.config_dir / 'config.yaml'} for settings")
        
        print("\n2. Test the system:")
        print("   python scripts/run_backtest.py  # Run backtest")
        print("   python scripts/train_models.py   # Train models")
        print("   python -m pytest tests/          # Run tests")
        
        print("\n3. Start the application:")
        print("   python main.py                   # Start trading")
        print("   docker-compose up               # Start with Docker")
        
        print("\n4. Access the dashboard:")
        print("   Web interface: http://localhost:8000")
        print("   Metrics: http://localhost:9090")
        
        if self.args.dev:
            print("\nDevelopment tools:")
            print("   black .                       # Format code")
            print("   flake8                        # Lint code")
            print("   pre-commit install            # Install git hooks")
        
        print("\n" + "="*60)
        print("IMPORTANT: Never commit API keys or .env file to version control!")
        print("="*60)
    
    def run(self) -> bool:
        """Run the complete setup process"""
        self.logger.info("Starting Bitcoin Trading AI Setup")
        self.logger.info(f"Project root: {self.project_root}")
        
        steps = []
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Create directories
        steps.append(("Creating directories", self.create_directories))
        
        # Install dependencies
        if self.args.install_deps:
            steps.append(("Installing dependencies", self.install_dependencies))
        
        # Create configuration
        if self.args.create_config:
            steps.append(("Creating configuration", self.create_configuration_files))
        
        # Initialize database
        if self.args.init_db:
            steps.append(("Initializing database", self.initialize_database))
        
        # Setup git hooks (development only)
        if self.args.dev:
            steps.append(("Setting up git hooks", self.setup_git_hooks))
        
        # Run tests
        if self.args.run_tests:
            steps.append(("Running tests", self.run_tests))
        
        # Create Docker files
        if self.args.create_docker:
            steps.append(("Creating Docker files", self.create_docker_files))
        
        # Create utility scripts
        if self.args.create_scripts:
            steps.append(("Creating utility scripts", self.create_utility_scripts))
        
        # Execute all steps
        for step_name, step_func in steps:
            self.logger.info(f"\n{step_name}...")
            if not step_func():
                self.logger.error(f"Failed: {step_name}")
                if not self.args.force:
                    return False
        
        # Print summary
        self.print_setup_summary()
        
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Bitcoin Trading AI Application Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete setup
  python setup.py --install-deps --create-config --init-db --run-tests
  
  # Development setup
  python setup.py --dev --install-deps --create-config
  
  # Docker setup
  python setup.py --create-docker --create-config
  
  # Minimal setup
  python setup.py --install-deps
        """
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install Python dependencies'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create configuration files'
    )
    
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database schema'
    )
    
    parser.add_argument(
        '--run-tests',
        action='store_true',
        help='Run test suite after setup'
    )
    
    parser.add_argument(
        '--create-docker',
        action='store_true',
        help='Create Docker configuration files'
    )
    
    parser.add_argument(
        '--create-scripts',
        action='store_true',
        help='Create utility scripts'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Setup for development (git hooks, etc.)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force operations (overwrite existing files)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all setup steps (excluding Docker)'
    )
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # If --all flag is used, enable all common setup steps
    if args.all:
        args.install_deps = True
        args.create_config = True
        args.init_db = True
        args.run_tests = True
        args.create_scripts = True
    
    return args


def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        setup_manager = SetupManager(args)
        success = setup_manager.run()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()