#!/bin/bash

# Setup data symlink for neuro-nav-vlm
# This creates a symlink to the neuro-nav data directory

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║            Data Directory Setup for neuro-nav-vlm                ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

SOURCE_DATA="/home/nick/Project_dir/neuro-nav/data"
TARGET_DATA="./data"

# Check if source exists
if [ ! -d "$SOURCE_DATA" ]; then
    echo "✗ Error: Source data directory not found at:"
    echo "  ${SOURCE_DATA}"
    echo ""
    echo "Make sure neuro-nav data exists first."
    exit 1
fi

# Check if target already exists
if [ -e "$TARGET_DATA" ]; then
    if [ -L "$TARGET_DATA" ]; then
        CURRENT_TARGET=$(readlink -f "$TARGET_DATA")
        SOURCE_RESOLVED=$(readlink -f "$SOURCE_DATA")
        
        if [ "$CURRENT_TARGET" = "$SOURCE_RESOLVED" ]; then
            echo "✓ Data symlink already correctly configured"
            echo "  Points to: ${SOURCE_RESOLVED}"
            exit 0
        else
            echo "⚠  Data symlink exists but points to:"
            echo "  ${CURRENT_TARGET}"
            echo ""
            read -p "Remove and recreate? (y/n) [y]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
                echo "Aborted."
                exit 0
            fi
            rm "$TARGET_DATA"
        fi
    elif [ -d "$TARGET_DATA" ]; then
        echo "⚠  Directory 'data' already exists (not a symlink)"
        echo ""
        read -p "Remove and create symlink? (y/n) [n]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted. Using existing data directory."
            exit 0
        fi
        rm -rf "$TARGET_DATA"
    else
        echo "⚠  File 'data' exists but is not a directory or symlink"
        read -p "Remove and create symlink? (y/n) [n]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
        rm "$TARGET_DATA"
    fi
fi

# Create symlink
echo "Creating symlink..."
ln -s "$SOURCE_DATA" "$TARGET_DATA"

if [ $? -eq 0 ]; then
    echo "✓ Symlink created successfully"
    echo ""
    echo "Data structure:"
    echo "  neuro-nav-vlm/data -> ${SOURCE_DATA}"
    echo ""
    echo "You can now access:"
    ls -la "$TARGET_DATA" 2>/dev/null | head -10
    echo ""
    echo "✓ Setup complete!"
else
    echo "✗ Failed to create symlink"
    exit 1
fi

