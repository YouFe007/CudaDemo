#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

# Check if nvcc can be found, throw an error if not
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc command not found."
    exit 1
fi

cd build
cmake .. && make