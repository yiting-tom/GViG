#!/bin/bash
# helper_functions.sh

# Helper function to check if a file exists
assert_file_exists() {
    if [ ! -f "$1" ]; then
        echo "File '$1' not found!"
        exit 1
    fi
}

# Function to display experiment settings (Generic Version)
display_exp_settings() {
    echo "-------------------------------- Experiment settings -------------------------------------"
    # Check if each variable is set, then print
    [ ! -z "${exp_tag+x}" ] && echo     "              Experiment tag: '${exp_tag}'"
    [ ! -z "${trainP+x}" ] && echo      "          Training prompt id: '${trainP}'"
    [ ! -z "${valP+x}" ] && echo        "        Validation prompt id: '${valP}'"
    [ ! -z "${testP+x}" ] && echo       "              Test prompt id: '${testP}'"
    echo "------------------------------------ File paths ------------------------------------------"
    [ ! -z "${train_data+x}" ] && echo  "     Training data file path: '${train_data}'"
    [ ! -z "${val_data+x}" ] && echo    "   Validation data file path: '${val_data}'"
    [ ! -z "${test_data+x}" ] && echo   "         Test data file path: '${test_data}'"
    [ ! -z "${save_path+x}" ] && echo   "        Checkpoint file path: '${save_path}'"
    [ ! -z "${log_path+x}" ] && echo    "               Log file path: '${log_path}'"
    [ ! -z "${result_name+x}" ] && echo "            Result file path: '${result_dir}/${result_name}_predict.json'"
    echo "------------------------------------------------------------------------------------------"
}

: '
process_and_split

Description:
    This function takes the name of a processing function and a list of file paths as arguments.
    It concatenates the content of all provided files, processes the concatenated content using 
    the specified processing function, and then splits the processed content back into individual
    files with the same order as provided.

Usage:
    process_and_split <processing_function_name> <file1> <file2> ... <fileN>

Parameters:
    processing_function_name: The name of the function to use for processing the concatenated content.
                              This function should accept a single argument: the path of the concatenated file.
                              It should produce the processed content to stdout.
    
    file1, file2, ... , fileN: Paths to the files to be concatenated and processed.

Output:
    This function will produce N output files, where N is the number of input files.
    Each output file will have the original filename appended with "_output".

Example:
    process_and_split sample_processor "file1.txt" "file2.txt" "file3.txt"

Note:
    Ensure that the processing function used does not modify the line count of the concatenated content,
    as the function splits processed content based on original line counts.
'
process_and_split() {
    local process_func="$1"
    shift  # Shift arguments to process the rest of the passed arguments
    local files=("$@")  # Store all remaining arguments as an array

    declare -a line_counts
    total_lines=0

    # Detect the number of lines for each file
    for file in "${files[@]}"; do
        lines=$(wc -l < "$file")
        line_counts+=($lines)
        total_lines=$(($total_lines + $lines))
    done

    # Concatenate the files
    cat "${files[@]}" > concatenated_file

    # Process the concatenated file using the provided function
    $process_func concatenated_file > processed_file

    # Split the processed file back into individual files
    start_line=1
    for i in "${!files[@]}"; do
        end_line=$(($start_line + ${line_counts[$i]} - 1))
        sed -n "$start_line,${end_line}p" processed_file > "${files[$i]}_output"
        start_line=$(($end_line + 1))
    done
}
