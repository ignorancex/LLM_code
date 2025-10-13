#!/usr/bin/env python3
"""
Context retrieval script for LLM agents.

Usage: python context.py [options]
Options:
  --branch <name>     Get specific branch context and recent commits
  --commit <hash>     View specific commit details  
  --log [--lines N]   View recent execution log entries
  --metadata <segment> Fetch specific metadata segment
"""

import sys
import json
import argparse
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(str(Path(__file__).parent))

from llm_agent_interface import LLMAgentInterface


def main():
    parser = argparse.ArgumentParser(description='Retrieve GCC project context')
    parser.add_argument('--branch', type=str, help='Get specific branch context')
    parser.add_argument('--commit', type=str, help='View specific commit details')
    parser.add_argument('--log', action='store_true', help='View recent execution log')
    parser.add_argument('--lines', type=int, default=20, help='Number of log lines to show (default: 20)')
    parser.add_argument('--metadata', type=str, help='Fetch specific metadata segment')
    args = parser.parse_args()
    
    # Initialize interface
    agent = LLMAgentInterface()
    
    # Build options based on arguments
    options = {}
    
    if args.branch:
        options["branch"] = args.branch
    elif args.commit:
        options["commit"] = args.commit
    elif args.log:
        options["log"] = True
        options["lines"] = args.lines
    elif args.metadata:
        options["metadata"] = args.metadata
    
    # Execute context command (does not pause - returns data directly)
    print("Retrieving context...")
    result = agent.show_context(options)
    
    # Format and display the result
    print("Context Result:")
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    main()