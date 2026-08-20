#!/bin/bash
# Diagnostic script to check disk space and permissions for checkpoint saving

echo "=== Disk Space Check ==="
df -h

echo ""
echo "=== Output Directory Check ==="
OUTPUT_DIR="${1:-~/dakota_rl_training/outputs/grpo_30b}"
echo "Checking: $OUTPUT_DIR"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Directory exists"
    ls -lah "$OUTPUT_DIR" | head -20
    echo ""
    echo "Weights directory:"
    if [ -d "$OUTPUT_DIR/weights" ]; then
        ls -lah "$OUTPUT_DIR/weights" | head -20
        echo ""
        echo "Disk usage of weights directory:"
        du -sh "$OUTPUT_DIR/weights"
    fi
else
    echo "Directory does not exist"
fi

echo ""
echo "=== Permission Check ==="
if [ -d "$OUTPUT_DIR" ]; then
    echo "Permissions:"
    ls -ld "$OUTPUT_DIR"
    echo ""
    echo "Can write test file?"
    touch "$OUTPUT_DIR/.write_test" 2>&1
    if [ $? -eq 0 ]; then
        echo "[OK] Write permission OK"
        rm "$OUTPUT_DIR/.write_test"
    else
        echo "[FAILED] Write permission FAILED"
    fi
fi

echo ""
echo "=== Inode Check ==="
df -i

echo ""
echo "=== Recent Checkpoint Files ==="
if [ -d "$OUTPUT_DIR/weights" ]; then
    find "$OUTPUT_DIR/weights" -name "*.tmp" -o -name "*.safetensors" -o -name "*.bin" | head -10
fi

