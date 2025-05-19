#!/bin/bash

# Set default directory using $HOME environment variable and parameter expansion
ROOT_DIR="${1:-$HOME/Projects/PoissonVAE/models}"

# Check if ROOT_DIR is actually the --dry-run flag
DRY_RUN=""
if [ "$ROOT_DIR" == "--dry-run" ] || [ "$ROOT_DIR" == "--dry_run" ]; then
    DRY_RUN="--dry-run"
    ROOT_DIR="$HOME/Projects/PoissonVAE/models"
    echo "No directory specified, using default: $ROOT_DIR"
else
    # Check for dry run as second parameter
    if [ "$2" == "--dry-run" ] || [ "$2" == "--dry_run" ]; then
        DRY_RUN="--dry-run"
    fi
fi

# Check if root directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory '$ROOT_DIR' does not exist"
    exit 1
fi

# Output dry run status if enabled
if [ -n "$DRY_RUN" ]; then
    echo "Running in dry-run mode. No files will be deleted."
fi

# Check if cleanup_chkpts.sh exists and is executable
CLEANUP_SCRIPT="./cleanup_chkpts.sh"
if [ ! -x "$CLEANUP_SCRIPT" ]; then
    echo "Error: cleanup_chkpts.sh not found or not executable"
    echo "Make sure cleanup_chkpts.sh is in the current directory and has execute permissions"
    exit 1
fi

# Create a temp file to store results
results_file=$(mktemp)

# Find all directories containing at least one .pt file
echo "Searching for directories containing .pt files..."
found_dirs=$(find "$ROOT_DIR" -type f -name "*.pt" -exec dirname {} \; | sort -u)

if [ -z "$found_dirs" ]; then
    echo "No directories containing .pt files found."
    rm "$results_file"
    exit 0
fi

# Count how many directories we found
dir_count=$(echo "$found_dirs" | wc -l)
echo "Found $dir_count directories containing .pt files."
echo

# Prepare results text file
hostname=$(hostname)
txt_file="./cleanup_results_${hostname}_$(date +%Y%m%d_%H%M%S).txt"
true > "$txt_file"  # Create/truncate the file using 'true' as a no-op command

# Process each directory
current=0
while IFS= read -r dir; do
    current=$((current + 1))
    echo "[$current/$dir_count] Processing: $dir"

    # Run cleanup_chkpts.sh on the directory
    "$CLEANUP_SCRIPT" "$dir" $DRY_RUN > /tmp/cleanup_output.txt

    # Extract kept file from output
    kept_file=$(grep "Keeping\|Would keep" /tmp/cleanup_output.txt | head -1 | sed 's/.*Keeping: //;s/.*Would keep: //')

    # If we found a kept file, add it to results
    if [ -n "$kept_file" ]; then
        echo "  Kept file: $kept_file"

        # Write directly to the results file
        {
            echo "$dir"
            echo "---------- Kept file: $kept_file"
            echo ""  # Add blank line between entries
        } >> "$txt_file"
    else
        echo "  Warning: Could not determine which file was kept"

        # Write directly to the results file
        {
            echo "$dir"
            echo "---------- Kept file: UNKNOWN"
            echo ""  # Add blank line between entries
        } >> "$txt_file"
    fi

    echo
done <<< "$found_dirs"

echo "Results saved to $txt_file"

# Clean up
rm -f /tmp/cleanup_output.txt

echo "Done!"
