#!/bin/bash
set -e

echo "================================================================="
echo "  Neuro-Nav GPU Jobs Smoke Test"
echo "================================================================="
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "================================================================="

echo ""
echo "[1] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found. Is this a GPU instance?"
fi

echo ""
echo "[2] Environment Variables:"
printenv | grep -E 'DATA_ROOT|OUTPUT_ROOT|CKPT_DIR|AWS|SCENE_ID' | sort

echo ""
echo "[3] Checking File System Mounts:"
df -h

echo ""
echo "[4] Checking Input Data Access (DATA_ROOT=${DATA_ROOT}):"
if [ -d "${DATA_ROOT}" ]; then
    echo "Directory exists."
    ls -F "${DATA_ROOT}" | head -n 10
else
    echo "ERROR: DATA_ROOT directory not found at ${DATA_ROOT}"
fi

echo ""
echo "[5] Checking Checkpoints Access (CKPT_DIR=${CKPT_DIR}):"
if [ -d "${CKPT_DIR}" ]; then
    echo "Directory exists."
    ls -F "${CKPT_DIR}" | head -n 10
else
    echo "WARNING: CKPT_DIR directory not found at ${CKPT_DIR}"
fi

echo ""
echo "[6] Testing Write Access to Output (OUTPUT_ROOT=${OUTPUT_ROOT}):"
if [ -d "${OUTPUT_ROOT}" ]; then
    TEST_FILE="${OUTPUT_ROOT}/smoke_test_$(date +%s).txt"
    echo "Attempting to write to ${TEST_FILE}..."
    echo "Smoke test run at $(date)" > "${TEST_FILE}"
    if [ -f "${TEST_FILE}" ]; then
        echo "SUCCESS: File written successfully."
        cat "${TEST_FILE}"
        # Optional: clean up
        # rm "${TEST_FILE}" 
    else
        echo "ERROR: Failed to write file."
    fi
else
    echo "ERROR: OUTPUT_ROOT directory not found at ${OUTPUT_ROOT}"
fi

echo ""
echo "================================================================="
echo "  Smoke Test Complete"
echo "================================================================="

