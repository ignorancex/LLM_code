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
# nsphere_distributions.py
"""
Wrapper script to execute the 1D variable distributions feature
of nsphere_plot.py using subprocess. This script effectively replaces
the older standalone generate_1d_comparison_histograms.py.
"""

import subprocess
import sys
import os
import argparse

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
    
    # First check if this is the root executable (./nsphere_distributions)
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

# --- Try to import read_lastparams to get default suffix ---
try:
    from nsphere_plot import read_lastparams
    DEFAULT_SUFFIX_FUNC = lambda: read_lastparams(return_suffix=True)
    # Optional: print import success only if verbose/debug later?
    # print("Successfully imported read_lastparams from nsphere_plot.")
except ImportError:
    # Don't print warning unless necessary (e.g., if suffix fails)
    DEFAULT_SUFFIX_FUNC = lambda: "_DEFAULT_SUFFIX_IMPORT_FAILED"
except Exception as e:
    DEFAULT_SUFFIX_FUNC = lambda: "_DEFAULT_SUFFIX_IMPORT_ERROR"

def run_script():
    """
    Main function that handles argument parsing, suffix determination,
    and subprocess execution of nsphere_plot.py with the distributions flag.
    """
    # --- Argument Parsing for the Wrapper ---
    parser = argparse.ArgumentParser(
        prog=invoked_name, #<-- Use the invoked name here for help messages
        description=f'Runs the 1D variable distributions generation from nsphere_plot.py.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--suffix', type=str,
                        help='Suffix for data files. If omitted, attempts to read from lastparams.dat.')
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
        sys.executable,        # Use the same Python interpreter
        target_script_path,    # Use absolute path
        "--distributions",     # Flag to run only this specific part
        "--suffix",
        run_suffix            # Pass the determined suffix
    ]

    # Add optional flags if they were provided to the wrapper
    if args.log:
        command.append("--log")

    # --- Execute the command ---
    try:
        # Run nsphere_plot.py, allow its output to go to console directly
        process = subprocess.run(command, check=True, text=True, stderr=sys.stderr, stdout=sys.stdout)
        # Exit with the same code as the subprocess if successful (usually 0)
        sys.exit(process.returncode)

    except FileNotFoundError:
        # More specific error if python itself isn't found vs target script
        if not os.path.exists(sys.executable):
             print(f"Error: Python executable not found at '{sys.executable}'.", file=sys.stderr)
        else:
            # Target script existence checked earlier, this shouldn't happen often
             print(f"Error: Could not execute command. Python or target script issue.", file=sys.stderr)
             print(f"Command: {' '.join(command)}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # nsphere_plot.py already printed its error message to stderr/stdout
        # print(f"\n{'='*80}", file=sys.stderr)
        # print(f"Error: Target script '{os.path.basename(target_script_path)}' failed with exit code {e.returncode}.", file=sys.stderr)
        # print(f"{'='*80}", file=sys.stderr)
        sys.exit(e.returncode) # Exit with the same error code
    except Exception as e:
         # Catch-all for unexpected wrapper errors
         print(f"\n{'='*80}", file=sys.stderr)
         print(f"An unexpected error occurred in '{invoked_name}': {e}", file=sys.stderr)
         print(f"{'='*80}", file=sys.stderr)
         sys.exit(1)


# Only execute the script if run directly
if __name__ == "__main__":
    run_script()