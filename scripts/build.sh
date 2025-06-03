#!/bin/bash
# Exit on any error to ensure robust execution
set -e

# Define ANSI color codes for colored output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

chmod +x ./scripts/init.sh
source ./scripts/init.sh

# Clean up previous build artifacts (except dist/)
echo -e "${YELLOW}Cleaning previous build artifacts...${NC}"
rm -rf build *.egg-info

# Build the extension and output directly to dist/
echo -e "${YELLOW}Building extension into dist/...${NC}"
python3 setup.py build_ext --build-lib ./

# Clean up residual files
echo -e "${YELLOW}Cleaning up residual files...${NC}"
rm -rf build *.egg-info

# Verify the module can be imported
echo -e "${YELLOW}Verifying module import...${NC}"
if python3 -c "from custom_cnn.cuda import _base; print('${GREEN}Successfully imported custom_cnn.cuda._base${NC}')"; then
    echo -e "${GREEN}Build and setup complete!${NC}"
else
    echo -e "${RED}Failed to import custom_cnn.cuda._base${NC}"
    exit 1
fi