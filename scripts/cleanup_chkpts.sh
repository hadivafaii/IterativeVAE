#!/bin/bash


# Check if directory argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory> [--dry-run]"
    echo "  <directory>  - Directory containing the files to process"
    echo "  --dry-run    - Optional flag to show what would be deleted without actually deleting"
    exit 1
fi

# Directory where the files are located
DIR="${1}"

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

# Check for dry run option
DRY_RUN=0
if [ "$2" == "--dry-run" ] || [ "$2" == "--dry_run" ]; then
    DRY_RUN=1
    echo "Running in dry-run mode. No files will be deleted."
fi

# Find all .pt files and store in a temporary file to handle spaces and special chars
temp_file=$(mktemp)
find "$DIR" -type f -name "*.pt" -print0 > "$temp_file"

if [ ! -s "$temp_file" ]; then
    echo "No .pt files found in directory '$DIR'"
    rm "$temp_file"
    exit 0
fi

# Create associative array to group files by their prefix pattern
declare -A FILE_GROUPS

# Process each .pt file to extract the number and group by prefix
while IFS= read -r -d $'\0' file; do
    # Extract just the filename without path
    filename=$(basename "$file")

    # Extract the prefix (everything before the first number)
    prefix=$(echo "$filename" | sed -E 's/([^0-9]*)([0-9]+).*/\1/')

    # Extract the first number in the filename
    number=$(echo "$filename" | grep -o -E '[0-9]+' | head -1)

    if [ -n "$number" ]; then
        # Add to the appropriate group
        if [ -z "${FILE_GROUPS[$prefix]}" ]; then
            FILE_GROUPS[$prefix]="$file"$'\n'"$number"
        else
            FILE_GROUPS[$prefix]="${FILE_GROUPS[$prefix]}"$'\n'"$file"$'\n'"$number"
        fi
    fi
done < "$temp_file"

rm "$temp_file"

# Process each group to find and keep only the highest numbered file
for prefix in "${!FILE_GROUPS[@]}"; do
    echo "Processing group with prefix: $prefix"

    highest_num=0
    highest_file=""

    # Convert the newline-separated list into an array
    readarray -t file_info_array <<< "${FILE_GROUPS[$prefix]}"

    # Create a sortable array of "number filename" pairs
    declare -a sort_array
    for ((i=0; i<${#file_info_array[@]}; i+=2)); do
        file="${file_info_array[i]}"
        num="${file_info_array[i+1]}"

        # Make sure to convert numbers with leading zeros properly
        # Remove any leading zeros to avoid octal interpretation issues
        num_int=$((10#$num))

        # Keep track of the highest file
        if [ "$num_int" -gt "$highest_num" ]; then
            highest_num="$num_int"
            highest_file="$file"
        fi

        # Add to sort array with padded numbers for proper sorting
        # Use a non-numeric padding approach for sorting
        padded_num=$(printf "%010s" "$num_int" | tr ' ' '0')
        sort_array+=("$padded_num $file")
    done

    # Sort the array and use mapfile to store result
    # First create a temp file from the array data
    temp_sort=$(mktemp)
    printf "%s\n" "${sort_array[@]}" > "$temp_sort"

    # Sort the temp file and read it into the sorted array
    mapfile -t sorted_array < <(sort "$temp_sort")

    # Clean up temp file
    rm "$temp_sort"

    echo "Highest numbered file in this group is: $(basename "$highest_file") with number $highest_num"

    # Process all files in ascending order
    for entry in "${sorted_array[@]}"; do
        # Extract file path (everything after the first space)
        file="${entry#* }"

        if [ "$file" != "$highest_file" ]; then
            if [ $DRY_RUN -eq 0 ]; then
                echo "  Deleting: $(basename "$file")"
                rm "$file"
            else
                echo "  Would delete: $(basename "$file")"
            fi
        else
            if [ $DRY_RUN -eq 0 ]; then
                echo "  Keeping: $(basename "$file")"
            else
                echo "  Would keep: $(basename "$file")"
            fi
        fi
    done

    echo ""
done

if [ $DRY_RUN -eq 0 ]; then
    echo "Done! Processed all .pt file groups."
else
    echo "Dry run complete! No files were deleted."
    echo "Run without --dry-run to actually delete files."
fi
