'''Utility functtions used across scripts'''


import os

def get_latest_file(directory, prefix, extension):
    """
    Used to retrieve files saved by date. When we don't know the exact filename but we know the prefix and extension.
    Get the latest file in the specified directory with the given prefix and extension.

    Args:
    - directory (str): The directory where to look for the files.
    - prefix (str): The prefix of the filenames to look for.
    - extension (str): The file extension to look for.

    Returns:
    - str: The path to the latest file.
    """
    # Get a list of files in the directory
    files = os.listdir(directory)

    # Filter for only the files that match the pattern
    files = [f for f in files if f.startswith(prefix) and f.endswith(extension)]

    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' and extension '{extension}' in directory '{directory}'")

    # Sort the files by date and time
    files.sort(reverse=True)

    # Get the latest file
    latest_file = files[0]
    return os.path.join(directory, latest_file)
