#!/bin/bash

# Check if the current directory is a valid git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "This is not a valid git repository"
    exit 1
fi

while true; do
    git pull
    sleep 2
done
