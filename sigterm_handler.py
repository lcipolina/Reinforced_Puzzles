''' Whenever a SIGTERM is recorded, it triggers these scripts
    For Tune, the time given by SIGTERM is not enough to save a checkpoint
    so this script just terminates Ray orderly.
'''

import signal
import sys
import os

# Use a relative path for the state file, which creates it in the current working directory
state_file_name = "experiment_state.txt"
state_file_path = os.path.join(os.getcwd(), state_file_name)

def return_state_file_path():
    return  os.path.join(os.getcwd(), state_file_name)

def signal_handler(sig, frame):
    print("SIGTERM received")
    # Note: Actual immediate checkpoint saving here may be complex due to the asynchronous nature.
    # Typically, you ensure that the training iteration completes and saves the state if possible.
    with open(state_file_path, "w") as f: # create 'experiment_state.txt' in the current working directory
        f.write("interrupted")
    sys.exit(0)




'''
def find_latest_checkpoint(checkpoint_dir):
   #Returns the directory of the latest checkpoint.
    checkpoint_paths = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)]
    checkpoint_dirs = [d for d in checkpoint_paths if os.path.isdir(d) and "checkpoint_" in os.path.basename(d)]
    if not checkpoint_dirs:
        return None
    latest_checkpoint_dir = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_checkpoint_dir
'''
