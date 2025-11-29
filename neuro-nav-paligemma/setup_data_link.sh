#!/bin/bash
# Create a symbolic link from neuro-nav data to neuro-nav-paligemma data

echo "Setting up data directory link..."

# Check if neuro-nav data exists
if [ ! -d "../neuro-nav/data" ]; then
    echo "❌ Error: ../neuro-nav/data not found"
    echo "Make sure neuro-nav directory exists and has run SLAM"
    exit 1
fi

# Remove existing data directory or link if it exists
if [ -e "data" ]; then
    if [ -L "data" ]; then
        echo "Removing existing symlink..."
        rm data
    else
        echo "❌ Error: 'data' already exists and is not a symlink"
        echo "Please remove or rename it manually"
        exit 1
    fi
fi

# Create symlink
ln -s ../neuro-nav/data data

echo "✅ Symbolic link created: data -> ../neuro-nav/data"
echo ""
echo "Now you can access the same SLAM outputs from all pipelines:"
echo "  neuro-nav/data/..."
echo "  neuro-nav-paligemma/data/... (same files)"

