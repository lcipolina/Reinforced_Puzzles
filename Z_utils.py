'''Utility functtions used across scripts'''


import os
import json


def my_print(msg, DEBUG = False):
    '''Toggle printing ON/OFF for debugging purposes'''
    if DEBUG:
        print(msg)

def get_latest_file_path(directory, name_prefix, extension):
    """
    Used to retrieve files saved by date. When we don't know the exact filename but we know the prefix and extension.
    Get the latest file in the specified directory with the given prefix and extension.

    Args:
    - directory (str): The directory where to look for the files.
    - name_prefix (str): The nameof the filenames to look for (before the date and time stamp)
    - extension (str): The file extension to look for.

    Returns:
    - str: The path to the latest file.
    """
    # Get a list of files in the directory
    files = os.listdir(directory)

    # Filter for only the files that match the pattern
    files = [f for f in files if f.startswith(name_prefix) and f.endswith(extension)]

    if not files:
        raise FileNotFoundError(f"No files found with name '{name_prefix}' and extension '{extension}' in directory '{directory}'")

    # Sort the files by date and time
    files.sort(reverse=True)

    # Get the latest file
    latest_file = files[0]
    return os.path.join(directory, latest_file)


def get_checkpoint_from_json_file(directory, name_prefix, extension):
    """
    Reads JSON data from a file and returns one of the checkpoints.

    Returns:
    str: One of the checkpoints.
    """
    file_path = get_latest_file_path(directory, name_prefix, extension)

    with open(file_path, 'r') as file:
        parsed_data = json.load(file)
    # Return one of the checkpoints
    return parsed_data[0]['best_checkpoint']
