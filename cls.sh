#!/bin/bash

# Delete all .png files in the current directory
rm -v *.png 2>/dev/null || echo "No .png files found to delete."
