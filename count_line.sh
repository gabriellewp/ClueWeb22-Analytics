#!/bin/bash

# Folder containing the files
folder_path="/ivi/ilps/datasets/MSMarco-Web-Search/100M/cw22_en_53M"
total_record=0

# Loop through each file in the folder
for file in "$folder_path"/*; do
  # Check if it's a regular file
  if [[ -f "$file" ]]; then
    # Count lines in the file
    line_count=$(wc -l < "$file")
    
    # Accumulate the line count into total_record
    total_record=$((total_record + line_count))
    
    # Check if the line count is greater than 1
    # if (( line_count > 1 )); then
    #   echo "File '$file' contains more than 1 record ($line_count lines)"
    # fi
  fi
done

# Print the total record count
echo "Total number of records: $total_record"
