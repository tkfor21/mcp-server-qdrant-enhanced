#!/usr/bin/env bash

# Enhanced Qdrant MCP Server Setup Script
# Provides dual installation options: NPM package or Docker container
# Part of @triepod-ai/mcp-server-qdrant-enhanced

set -euo pipefail

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="@triepod-ai/mcp-server-qdrant-enhanced"
# Local image tag built from source (the golden path is build-from-source; no
# prebuilt registry image is pulled — see install_docker_method).
DOCKER_IMAGE="mcp-server-qdrant-enhanced"
REQUIRED_PYTHON_VERSION="3.10"
REQUIRED_NODE_VERSION="18"

# Global variables
INSTALLATION_METHOD=""
QDRANT_URL=""
COLLECTION_NAME=""
MCP_CLIENT=""
CONFIG_DIR=""

# Utility functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "\n${BOLD}${CYAN}$1${NC}\n"
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

version_compare() {
    # Compare two version strings (returns 0 if $1 >= $2)
    python3 -c "
from packaging import version
import sys
try:
    result = version.parse('$1') >= version.parse('$2')
    sys.exit(0 if result else 1)
except:
    # Fallback for simple version comparison
    v1 = [int(x) for x in '$1'.split('.')]
    v2 = [int(x) for x in '$2'.split('.')]
    sys.exit(0 if v1 >= v2 else 1)
" 2>/dev/null
}

# Detection functions
detect_platform() {
    case "$(uname -s)" in
        Darwin*) echo "macOS" ;;
        Linux*) echo "Linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "Windows" ;;
        *) echo "Unknown" ;;
    esac
}

detect_architecture() {
    case "$(uname -m)" in
        x86_64|amd64) echo "amd64" ;;
        arm64|aarch64) echo "arm64" ;;
        *) echo "unknown" ;;
    esac
}

check_python() {
    if ! check_command python3; then
        return 1
    fi
    
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    if ! version_compare "$python_version" "$REQUIRED_PYTHON_VERSION"; then
        log_error "Python $REQUIRED_PYTHON_VERSION or later required, found $python_version"
        return 1
    fi
    
    return 0
}

check_node() {
    if ! check_command node; then
        return 1
    fi
    
    local node_version
    node_version=$(node --version | sed 's/v//')
    
    if ! version_compare "$node_version" "$REQUIRED_NODE_VERSION"; then
        log_error "Node.js $REQUIRED_NODE_VERSION or later required, found $node_version"
        return 1
    fi
    
    return 0
}

check_docker() {
    if ! check_command docker; then
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is installed but not running"
        return 1
    fi
    
    return 0
}

check_qdrant_connection() {
    local url="$1"
    log_info "Testing connection to Qdrant at $url..."
    
    if check_command curl; then
        if curl -s "$url/health" >/dev/null 2>&1; then
            log_success "Successfully connected to Qdrant"
            return 0
        fi
    elif check_command wget; then
        if wget -q --spider "$url/health" 2>/dev/null; then
            log_success "Successfully connected to Qdrant"
            return 0
        fi
    elif check_command python3; then
        if python3 -c "
import urllib.request
try:
    urllib.request.urlopen('$url/health', timeout=5)
    print('✅ Successfully connected to Qdrant')
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
            return 0
        fi
    fi
    
    log_warning "Could not connect to Qdrant at $url"
    log_info "Please ensure Qdrant is running and accessible"
    return 1
}

# Installation methods
install_npm_method() {
    log_header "Installing via NPM Package"
    
    # Check Node.js
    if ! check_node; then
        log_error "Node.js $REQUIRED_NODE_VERSION or later is required for NPM installation"
        log_info "Please install Node.js from https://nodejs.org/"
        return 1
    fi
    
    # Check npm
    if ! check_command npm; then
        log_error "npm is required but not found"
        return 1
    fi
    
    # Install the package
    log_info "Installing $PACKAGE_NAME..."
    if npm install -g "$PACKAGE_NAME"; then
        log_success "NPM package installed successfully"
    else
        log_error "Failed to install NPM package"
        return 1
    fi
    
    # Verify installation
    if check_command mcp-server-qdrant-enhanced; then
        log_success "Installation verified - mcp-server-qdrant-enhanced command is available"
    else
        log_warning "Command not found in PATH - you may need to restart your terminal"
    fi
    
    return 0
}

install_docker_method() {
    log_header "Installing via Docker Container"
    
    # Check Docker
    if ! check_docker; then
        log_error "Docker is required but not available"
        log_info "Please install Docker from https://docs.docker.com/get-docker/"
        return 1
    fi
    
    # Build the image from source so it matches this checkout (deps pinned via
    # uv.lock). No prebuilt registry image is pulled — build-from-source is the
    # golden path.
    log_info "Building Docker image $DOCKER_IMAGE:latest from source (may take a few minutes)..."
    if docker build -t "$DOCKER_IMAGE:latest" "$SCRIPT_DIR"; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        return 1
    fi

    log_info "To run the server as a persistent HTTP service (:10650) alongside a"
    log_info "local Qdrant, use the committed compose file:  docker compose up -d"
    return 0
}

# The repo ships a golden docker-compose.yml (Qdrant + server, build-from-source).
# Setup no longer generates or overwrites it — run the service with:
#   docker compose up -d

# Configuration generation
generate_claude_desktop_config() {
    local config_file="$1"
    local method="$2"
    
    log_info "Generating Claude Desktop configuration..."
    
    if [[ "$method" == "npm" ]]; then
        cat > "$config_file" << EOF
{
  "mcpServers": {
    "qdrant-enhanced": {
      "command": "mcp-server-qdrant-enhanced",
      "args": ["--transport", "stdio"],
      "env": {
        "QDRANT_URL": "$QDRANT_URL",
        "COLLECTION_NAME": "$COLLECTION_NAME"
      }
    }
  }
}
EOF
    else
        cat > "$config_file" << EOF
{
  "mcpServers": {
    "qdrant-enhanced": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--network", "host",
        "-e", "MCP_TRANSPORT=stdio",
        "-e", "QDRANT_URL=$QDRANT_URL",
        "-e", "COLLECTION_NAME=$COLLECTION_NAME",
        "$DOCKER_IMAGE:latest"
      ]
    }
  }
}
EOF
    fi
    
    log_success "Claude Desktop configuration saved to $config_file"
}

generate_vscode_config() {
    local config_file="$1"
    local method="$2"
    
    log_info "Generating VS Code MCP configuration..."
    
    if [[ "$method" == "npm" ]]; then
        cat > "$config_file" << EOF
{
  "mcp": {
    "servers": {
      "qdrant-enhanced": {
        "command": "mcp-server-qdrant-enhanced",
        "args": ["--transport", "stdio"],
        "env": {
          "QDRANT_URL": "$QDRANT_URL",
          "COLLECTION_NAME": "$COLLECTION_NAME"
        }
      }
    }
  }
}
EOF
    else
        cat > "$config_file" << EOF
{
  "mcp": {
    "servers": {
      "qdrant-enhanced": {
        "command": "docker",
        "args": [
          "run",
          "--rm",
          "-i",
          "--network", "host",
          "-e", "MCP_TRANSPORT=stdio",
          "-e", "QDRANT_URL=$QDRANT_URL",
          "-e", "COLLECTION_NAME=$COLLECTION_NAME",
          "$DOCKER_IMAGE:latest"
        ]
      }
    }
  }
}
EOF
    fi
    
    log_success "VS Code MCP configuration saved to $config_file"
}

# User interaction
prompt_installation_method() {
    log_header "Choose Installation Method"
    
    echo "Which installation method would you prefer?"
    echo
    echo "1) NPM Package (Recommended for development)"
    echo "   - Lightweight, integrates with existing Node.js tools"
    echo "   - Requires Node.js $REQUIRED_NODE_VERSION+ and Python $REQUIRED_PYTHON_VERSION+"
    echo "   - Direct command access: mcp-server-qdrant-enhanced"
    echo
    echo "2) Docker Container (Recommended for production)"
    echo "   - Complete isolation, no local dependencies"
    echo "   - Pre-built 16.5GB image with CUDA + cuDNN + models"
    echo "   - GPU acceleration ready"
    echo
    
    while true; do
        read -p "Enter your choice (1 or 2): " choice
        case $choice in
            1)
                INSTALLATION_METHOD="npm"
                break
                ;;
            2)
                INSTALLATION_METHOD="docker"
                break
                ;;
            *)
                log_warning "Please enter 1 or 2"
                ;;
        esac
    done
}

prompt_qdrant_config() {
    log_header "Qdrant Configuration"
    
    echo "Enter your Qdrant connection details:"
    echo
    
    # Qdrant URL
    read -p "Qdrant URL (default: http://localhost:6333): " url_input
    QDRANT_URL="${url_input:-http://localhost:6333}"
    
    # Collection name
    read -p "Default collection name (default: enhanced-collection): " collection_input
    COLLECTION_NAME="${collection_input:-enhanced-collection}"
    
    # Test connection
    check_qdrant_connection "$QDRANT_URL"
}

prompt_mcp_client() {
    log_header "MCP Client Configuration"
    
    echo "Which MCP client would you like to configure?"
    echo
    echo "1) Claude Desktop"
    echo "2) VS Code"
    echo "3) Both"
    echo "4) Skip configuration"
    echo
    
    while true; do
        read -p "Enter your choice (1-4): " choice
        case $choice in
            1)
                MCP_CLIENT="claude"
                break
                ;;
            2)
                MCP_CLIENT="vscode"
                break
                ;;
            3)
                MCP_CLIENT="both"
                break
                ;;
            4)
                MCP_CLIENT="none"
                break
                ;;
            *)
                log_warning "Please enter 1, 2, 3, or 4"
                ;;
        esac
    done
}

# Main installation flow
main() {
    log_header "🚀 Enhanced Qdrant MCP Server Setup"
    
    echo "Welcome to the Enhanced Qdrant MCP Server setup!"
    echo "This script will help you install and configure the production-ready"
    echo "MCP server with GPU acceleration and multi-vector support."
    echo
    
    # Detect platform
    local platform=$(detect_platform)
    local arch=$(detect_architecture)
    log_info "Detected platform: $platform ($arch)"
    
    # Prompt for installation method
    prompt_installation_method
    
    # Prompt for Qdrant configuration
    prompt_qdrant_config
    
    # Install based on chosen method
    case $INSTALLATION_METHOD in
        npm)
            if install_npm_method; then
                log_success "NPM installation completed successfully"
            else
                log_error "NPM installation failed"
                exit 1
            fi
            ;;
        docker)
            if install_docker_method; then
                log_success "Docker installation completed successfully"
            else
                log_error "Docker installation failed"
                exit 1
            fi
            ;;
    esac
    
    # Prompt for MCP client configuration
    prompt_mcp_client
    
    # Generate configurations
    if [[ "$MCP_CLIENT" != "none" ]]; then
        if [[ "$MCP_CLIENT" == "claude" || "$MCP_CLIENT" == "both" ]]; then
            generate_claude_desktop_config "claude_desktop_config.json" "$INSTALLATION_METHOD"
        fi
        
        if [[ "$MCP_CLIENT" == "vscode" || "$MCP_CLIENT" == "both" ]]; then
            generate_vscode_config "vscode_mcp_settings.json" "$INSTALLATION_METHOD"
        fi
    fi
    
    # Final instructions
    log_header "🎉 Setup Complete!"
    
    echo "Your Enhanced Qdrant MCP Server is ready to use!"
    echo
    
    case $INSTALLATION_METHOD in
        npm)
            echo "✨ NPM Package Installation:"
            echo "  Command: mcp-server-qdrant-enhanced"
            echo "  Alternative: qdrant-mcp-enhanced"
            echo
            ;;
        docker)
            echo "🐳 Docker Installation:"
            echo "  Start service: docker-compose up -d"
            echo "  Stop service: docker-compose down"
            echo "  View logs: docker-compose logs -f"
            echo
            ;;
    esac
    
    echo "🔧 Configuration:"
    echo "  Qdrant URL: $QDRANT_URL"
    echo "  Collection: $COLLECTION_NAME"
    echo
    
    if [[ "$MCP_CLIENT" != "none" ]]; then
        echo "📁 Generated configuration files:"
        [[ "$MCP_CLIENT" == "claude" || "$MCP_CLIENT" == "both" ]] && echo "  - claude_desktop_config.json"
        [[ "$MCP_CLIENT" == "vscode" || "$MCP_CLIENT" == "both" ]] && echo "  - vscode_mcp_settings.json"
        echo
    fi
    
    echo "🚀 Enhanced Features Available:"
    echo "  • GPU Acceleration with FastEmbed"
    echo "  • Multi-Vector Support (384D/768D/1024D)"
    echo "  • 48 Production Collection Configurations"
    echo "  • Automatic Model Selection"
    echo "  • Quantization Enabled"
    echo
    
    echo "📚 Next Steps:"
    echo "  1. Restart your MCP client (Claude Desktop, VS Code, etc.)"
    echo "  2. Import the generated configuration files"
    echo "  3. Test the connection with a simple query"
    echo
    
    log_success "Setup completed successfully! Enjoy your enhanced MCP server! 🎊"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi