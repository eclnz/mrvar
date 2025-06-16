#!/bin/bash

# Script to process anatomical MRI scans using process_variance.py
# Usage: ./downsample_anat.sh <bids_directory>

# Check if BIDS directory is provided
if [ $# -ne 1 ]; then
    echo "Error: Please provide the BIDS directory path"
    echo "Usage: ./downsample_anat.sh <bids_directory>"
    exit 1
fi

BIDS_DIR="$1"

# Validate BIDS directory
if [ ! -d "$BIDS_DIR" ]; then
    echo "Error: BIDS directory '$BIDS_DIR' does not exist"
    exit 1
fi

# Set up logging
log_file="downsample_anat.log"
echo "Starting anatomical MRI processing pipeline at $(date)" | tee -a "$log_file"
echo "Processing BIDS directory: $BIDS_DIR" | tee -a "$log_file"

# Create derivatives directory if it doesn't exist
mkdir -p "$BIDS_DIR/derivatives/aMRI"

# Loop through all subjects
for subject_dir in "$BIDS_DIR/raw/sub-"*; do
    if [ ! -d "$subject_dir" ]; then
        echo "Skipping $subject_dir - not a directory" | tee -a "$log_file"
        continue
    fi

    subject=$(basename "$subject_dir")
    echo "Processing $subject..." | tee -a "$log_file"

    # Loop through all sessions
    for session_dir in "$subject_dir"/ses-*; do
        if [ ! -d "$session_dir" ]; then
            echo "Skipping $session_dir - not a directory" | tee -a "$log_file"
            continue
        fi

        session=$(basename "$session_dir")
        echo "Processing $session..." | tee -a "$log_file"

        # Process anatomical scan
        anat_file="$session_dir/anat/${subject}_${session}_aMRI.nii.gz"
        
        if [ ! -f "$anat_file" ]; then
            echo "Warning: Anatomical scan not found at $anat_file" | tee -a "$log_file"
            continue
        fi

        # Create output directory
        output_dir="$BIDS_DIR/derivatives/aMRI/$subject/$session"
        mkdir -p "$output_dir"

        # Define output file path
        output_file="$output_dir/${subject}_${session}_raw_variance_downsampled.nii.gz"

        echo "Processing $anat_file to $output_file" | tee -a "$log_file"

        # Call process_variance.py with downsampling
        if python process_variance.py "$anat_file" "$output_file" --downsample 4; then
            echo "Successfully processed $anat_file" | tee -a "$log_file"
        else
            echo "Error: Failed to process $anat_file" | tee -a "$log_file"
            continue
        fi

    done
done

echo "Pipeline completed at $(date)" | tee -a "$log_file" 