#!/bin/bash

# Setup script for third-party dependencies
# This script downloads the required third-party libraries from DREAMPlace

echo "Setting up third-party dependencies from DREAMPlace..."

# Check if thirdparty directory already exists
if [ -d "thirdparty" ]; then
    echo "Warning: thirdparty directory already exists."
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    rm -rf thirdparty
fi

# Clone DREAMPlace repository
echo "Cloning DREAMPlace repository..."
git clone https://github.com/limbo018/DREAMPlace.git temp_dreamplace

if [ $? -ne 0 ]; then
    echo "Error: Failed to clone DREAMPlace repository."
    exit 1
fi

# Copy thirdparty directory
echo "Copying thirdparty directory..."
cp -r temp_dreamplace/thirdparty ./

# Clean up
echo "Cleaning up..."
rm -rf temp_dreamplace

echo "✓ Third-party dependencies setup completed!"
echo "The following libraries have been installed in ./thirdparty/:"
ls -1 thirdparty/

