#!/bin/bash

TEST_DIR="tests/"


TEST_LIST="tests/geoCosiCorr3D_tests.txt"

export LD_LIBRARY_PATH=$(pwd)/lib/:$LD_LIBRARY_PATH
if [ ! -f "$TEST_LIST" ]; then
    echo "Test list file $TEST_LIST does not exist."
    exit 1
fi

PYTEST_CMD="pytest"

while IFS= read -r test_script; do
    if [ -f "${TEST_DIR}${test_script}" ]; then
        PYTEST_CMD="$PYTEST_CMD ${TEST_DIR}${test_script}"
    else
        echo "Test script ${test_script} not found in ${TEST_DIR}"
    fi
done < "$TEST_LIST"

echo "Running tests with pytest..."
$PYTEST_CMD
echo "Tests completed."
