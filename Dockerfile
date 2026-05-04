# Bitcoin Trading AI Application Dockerfile
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder - Install dependencies and build Python packages
# ============================================================================
FROM python:3.9-slim AS builder

# Set build arguments
ARG BUILD_ENV=production
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    wget \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install TA-LIB from source (required for technical analysis)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker cache
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Create optimized production image
# ============================================================================
FROM python:3.9-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    PYTHONPATH=/app \
    # Security
    GUNICORN_USER=appuser \
    GUNICORN_GROUP=appgroup \
    # Performance
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Create non-root user for security
RUN groupadd -r $GUNICORN_GROUP && \
    useradd -r -g $GUNICORN_GROUP $GUNICORN_USER

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Database clients
    postgresql-client \
    # Redis
    redis-tools \
    # SSL
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    # Monitoring
    procps \
    htop \
    # Debugging
    vim \
    less \
    # Timezone data
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone (adjust as needed)
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
WORKDIR $APP_HOME
COPY . $APP_HOME/

# Create necessary directories
RUN mkdir -p \
    $APP_HOME/data/raw \
    $APP_HOME/data/processed \
    $APP_HOME/data/cache \
    $APP_HOME/models \
    $APP_HOME/logs \
    $APP_HOME/results \
    $APP_HOME/backups \
    && chown -R $GUNICORN_USER:$GUNICORN_GROUP $APP_HOME

# Fix permissions
RUN chmod -R 755 $APP_HOME/scripts/*.sh 2>/dev/null || true && \
    chmod -R 755 $APP_HOME/scripts/*.py 2>/dev/null || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; import socket; \
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); \
    result = sock.connect_ex(('localhost', 8000)); \
    sock.close(); \
    sys.exit(result == 0 or 1)"

# Switch to non-root user
USER $GUNICORN_USER

# Expose ports
EXPOSE 8000  # FastAPI web interface
EXPOSE 9090  # Prometheus metrics

# ============================================================================
# Stage 3: Development - Include development tools
# ============================================================================
FROM runtime AS development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Debugging tools
    gdb \
    strace \
    lsof \
    # Network tools
    net-tools \
    iputils-ping \
    dnsutils \
    # Build tools
    pkg-config \
    # Shell
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    ipython \
    ipdb \
    jupyter \
    jupyterlab \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    pre-commit \
    bandit \
    safety

# Create Jupyter config
RUN mkdir -p /home/$GUNICORN_USER/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" > /home/$GUNICORN_USER/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /home/$GUNICORN_USER/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/$GUNICORN_USER/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /home/$GUNICORN_USER/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/$GUNICORN_USER/.jupyter/jupyter_notebook_config.py && \
    chown -R $GUNICORN_USER:$GUNICORN_GROUP /home/$GUNICORN_USER/.jupyter

# Switch back to app user
USER $GUNICORN_USER

# Expose Jupyter port
EXPOSE 8888

# ============================================================================
# Stage 4: Production with GPU support (optional)
# ============================================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    PYTHONPATH=/app \
    # CUDA
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Security
    GUNICORN_USER=appuser \
    GUNICORN_GROUP=appgroup

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3.9-venv \
    postgresql-client \
    redis-tools \
    ca-certificates \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.9 -m venv /opt/venv

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r $GUNICORN_GROUP && \
    useradd -r -g $GUNICORN_GROUP $GUNICORN_USER

# Copy application code
WORKDIR $APP_HOME
COPY . $APP_HOME/

# Create necessary directories
RUN mkdir -p \
    $APP_HOME/data/raw \
    $APP_HOME/data/processed \
    $APP_HOME/data/cache \
    $APP_HOME/models \
    $APP_HOME/logs \
    $APP_HOME/results \
    $APP_HOME/backups \
    && chown -R $GUNICORN_USER:$GUNICORN_GROUP $APP_HOME

# Fix permissions
RUN chmod -R 755 $APP_HOME/scripts/*.sh 2>/dev/null || true

# Switch to non-root user
USER $GUNICORN_USER

# Expose ports
EXPOSE 8000 9090

# ============================================================================
# Common entrypoints and commands
# ============================================================================

# Default command (can be overridden)
CMD ["python", "main.py"]

# Entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]