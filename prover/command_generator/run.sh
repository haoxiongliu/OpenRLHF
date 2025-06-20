#!/bin/bash

# Lean Command Generator Web Service Launcher
# This script helps start the web service easily

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Lean Command Generator Web Service${NC}"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: app.py not found. Please run this script from the prover/command_generator directory.${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check if required packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import fastapi, uvicorn, jinja2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing required dependencies...${NC}"
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install dependencies.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Dependencies OK!${NC}"

# Start the server
echo -e "${GREEN}Starting server...${NC}"
echo "Press Ctrl+C to stop the server"
echo ""

# Parse command line arguments
HOST="0.0.0.0"
PORT="8000"
RELOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --host HOST     Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT     Port to bind to (default: 8000)"
            echo "  --reload        Enable auto-reload for development"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Server will be available at: http://${HOST}:${PORT}${NC}"
echo -e "${GREEN}API documentation at: http://${HOST}:${PORT}/api/docs${NC}"
echo ""

# Start the server
python3 start_server.py --host "$HOST" --port "$PORT" $RELOAD 