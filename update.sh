#!/bin/bash

git fetch origin main
git reset --hard origin/main

chmod +x examples/WanVSR/infer.py

echo "Successfully updated."
