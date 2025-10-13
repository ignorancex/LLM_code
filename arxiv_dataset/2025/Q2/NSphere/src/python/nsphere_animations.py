#!/usr/bin/env python3
# Copyright 2025 Kris Sigurdson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# nsphere_animations.py
"""
Wrapper script to execute the animation generation features (Phase Space,
Mass, Density, Psi) of nsphere_plot.py using subprocess.
"""

import subprocess
import sys
import os
import argparse
import glob
import re # Needed for finding max snapshot

# --- Project Root and Path Setup ---
def find_project_root():
    """
    Determines the project root directory based on script location.
    
    Returns
    -------
    str
        Absolute path to the project root directory
        
    Notes
    -----
    Uses multiple heuristics to reliably find the project root:
    1. Navigate up from script location to find the parent of src/
    2. Check for expected project structure indicators
    3. Check if running from project root with expected directories
    4. Falls back to current working directory as a last resort
    """
    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)
    script_dir = os.path.dirname(script_path)
    
    # First check if this is the root executable (./nsphere_animations)
    cwd = os.getcwd()
    if (os.path.isdir(os.path.join(cwd, 'src')) and 
        os.path.isdir(os.path.join(cwd, 'data')) and 
        os.path.isdir(os.path.join(cwd, 'results'))):
        return cwd  # We're running from the project root
    
    # Navigate up to find the directory containing 'src'
    potential_root = os.path.dirname(os.path.dirname(script_path)) # Go up two levels from src/python
    

    # Fallback if structure is different or running from unexpected location
    # Check if current script dir's parent is 'src'
    if os.path.basename(os.path.dirname(script_path)) == 'src':
        return os.path.dirname(os.path.dirname(script_path)) # Parent of src

    # Check if running from root where 'src' exists
    if os.path.isdir(os.path.join(os.path.dirname(script_path), 'src')):
        return os.path.dirname(script_path)

    # Default to CWD as a last resort
    print("Warning: Could not reliably determine project root from script location. Using CWD.", file=sys.stderr)
    return os.getcwd()

PROJECT_ROOT = find_project_root()

# Define standard runtime directory paths relative to root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOG_DIR = os.path.join(PROJECT_ROOT, 'log')
INIT_DIR = os.path.join(PROJECT_ROOT, 'init')

# Ensure Python can find modules within src/python if needed
python_src_dir = os.path.join(PROJECT_ROOT, 'src', 'python')
if python_src_dir not in sys.path:
    sys.path.insert(0, python_src_dir)
# --- End Project Root Setup ---

# --- Determine the name the script was run with ---
invoked_name = os.path.basename(sys.argv[0])

# --- Try to import read_lastparams from nsphere_plot to get default suffix ---
# --- Also need get_snapshot_number to find max snapshot for default end ---
try:
    from nsphere_plot import read_lastparams, get_snapshot_number
    DEFAULT_SUFFIX_FUNC = lambda: read_lastparams(return_suffix=True)
except ImportError:
    # Suppress warning unless suffix is actually needed and fails
    DEFAULT_SUFFIX_FUNC = None
    get_snapshot_number = None # Mark as unavailable
except Exception as e:
    # Suppress warning unless suffix is actually needed and fails
    DEFAULT_SUFFIX_FUNC = None
    get_snapshot_number = None
    # Optionally log the import error if logging was enabled for the wrapper?
    # print(f"Warning: Error during nsphere_plot import: {e}", file=sys.stderr)


# --- Function to find max snapshot (similar to original script) ---
def find_max_snap(suffix_to_find):
    if not suffix_to_find or not get_snapshot_number:
        return 0 # Cannot determine without suffix or helper function
    max_snap = 0
    # Use the Rank_Mass_Rad_VRad_sorted pattern as nsphere_plot does
    Rank_pattern = os.path.join(DATA_DIR, f"Rank_Mass_Rad_VRad_sorted_t*{suffix_to_find}.dat")
    # Regex needed for get_snapshot_number if using default pattern
    pattern = re.compile(r'Rank_Mass_Rad_VRad_sorted_t(\d+)')
    try:
        all_files = glob.glob(Rank_pattern)
        if all_files:
             # Filter precisely like nsphere_plot.py does
            correct_pattern = re.compile(re.escape(os.path.join(DATA_DIR, 'Rank_Mass_Rad_VRad_sorted_t')) + r'\d+' + re.escape(suffix_to_find) + r'\.dat$')
            filtered_files = [f for f in all_files if correct_pattern.match(f)]
            if filtered_files:
                 # Sort by snapshot number before getting max
                filtered_files.sort(key=lambda f: get_snapshot_number(f, pattern=pattern))
                # Use the last file after sorting
                max_snap = get_snapshot_number(filtered_files[-1], pattern=pattern)
                if max_snap == 999999999: max_snap = 0 # Handle non-match case from helper
    except Exception as e:
        # Suppress warning unless needed?
        # print(f"Warning: Error finding max snapshot number: {e}", file=sys.stderr)
        max_snap = 0
    return max_snap

def run_script():
    """
    Main function that handles argument parsing, suffix determination,
    and subprocess execution of nsphere_plot.py with the animations flag.
    """
    # --- Argument Parsing for the Wrapper ---
    parser = argparse.ArgumentParser(
        prog=invoked_name,
        description='Runs the animation generation features from nsphere_plot.py.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--suffix', type=str,
                        help='Suffix for data files. If omitted, attempts to read from lastparams.dat.')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting snapshot number for animations.')
    # Default end to 0, nsphere_plot handles 0 as 'auto/max'
    parser.add_argument('--end', type=int, default=0,
                        help='Ending snapshot number. Default 0 means use all available.')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size between snapshots.')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for output animations.')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging in nsphere_plot.py.')

    args = parser.parse_args()

    # --- Determine final suffix ---
    run_suffix = None
    if args.suffix is not None:
        # User provided suffix explicitly
        run_suffix = args.suffix
    else:
        # Suffix not provided, try to read default
        if DEFAULT_SUFFIX_FUNC: # Check if import function exists
            try:
                run_suffix = DEFAULT_SUFFIX_FUNC()
                if "_DEFAULT_" in run_suffix or "_ERROR" in run_suffix:
                    # Function returned an error indicator
                    print(f"Error: Could not determine default suffix from nsphere_plot.py/lastparams.dat.", file=sys.stderr)
                    run_suffix = None # Mark as failed
            except Exception as e:
                print(f"Error calling read_lastparams: {e}", file=sys.stderr)
                run_suffix = None # Mark as failed
        else:
            # Import failed earlier
            print(f"Error: Cannot read default suffix because import failed.", file=sys.stderr)
            run_suffix = None # Mark as failed

        # Exit if suffix determination failed
        if run_suffix is None:
            print(f"Please provide a valid suffix using --suffix.", file=sys.stderr)
            sys.exit(1)


    # --- Construct the command for nsphere_plot executable ---
    # Look for nsphere_plot in the PROJECT_ROOT directory
    target_script_name = "nsphere_plot"
    target_script_path = os.path.join(PROJECT_ROOT, target_script_name)
    
    # If not found in project root, check bin directory
    if not os.path.exists(target_script_path):
        target_script_path = os.path.join(PROJECT_ROOT, "bin", target_script_name)
        
    # If still not found, try the source file directly
    if not os.path.exists(target_script_path):
        target_script_path = os.path.join(PROJECT_ROOT, "src", "python", "nsphere_plot.py")
    
    if not os.path.exists(target_script_path):
        print(f"Error: The nsphere_plot executable was not found.", file=sys.stderr)
        print(f"Looked in: {os.path.join(PROJECT_ROOT, target_script_name)}", file=sys.stderr)
        print(f"       and: {os.path.join(PROJECT_ROOT, 'bin', target_script_name)}", file=sys.stderr)
        print(f"       and: {os.path.join(PROJECT_ROOT, 'src', 'python', 'nsphere_plot.py')}", file=sys.stderr)
        sys.exit(1)

    command = [
        sys.executable,
        target_script_path,
        "--animations",        # Flag to run only animation generation
        "--suffix", run_suffix
    ]

    # Add optional arguments only if they differ from nsphere_plot.py defaults
    if args.start != 0:
        command.extend(["--start", str(args.start)])
    # nsphere_plot uses 0 as default for end (meaning max available), so only pass if user specified non-zero
    if args.end != 0:
         command.extend(["--end", str(args.end)])
    if args.step != 1:
        command.extend(["--step", str(args.step)])
    if args.fps != 10:
        command.extend(["--fps", str(args.fps)])
    if args.log:
        command.append("--log")

    # --- Execute the command ---
    try:
        # Removed introductory wrapper messages
        process = subprocess.run(command, check=True, text=True, stderr=sys.stderr, stdout=sys.stdout)
        # Removed completion wrapper messages
        sys.exit(process.returncode)

    except FileNotFoundError:
        if not os.path.exists(sys.executable):
             print(f"Error: Python executable not found at '{sys.executable}'.", file=sys.stderr)
        else:
             print(f"Error: Could not execute command.", file=sys.stderr)
             print(f"Command: {' '.join(command)}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # nsphere_plot.py likely already printed its error message
        print(f"\nError: Target script '{os.path.basename(target_script_path)}' failed with exit code {e.returncode}.", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
         print(f"\nAn unexpected error occurred in '{invoked_name}': {type(e).__name__} - {e}", file=sys.stderr)
         sys.exit(1)

# Only execute the script if run directly
if __name__ == "__main__":
    run_script()