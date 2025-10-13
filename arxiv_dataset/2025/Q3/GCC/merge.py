#!/usr/bin/env python3
"""
Merge script for LLM agents.

Usage: python merge.py --branches branch1 branch2 [branch3 ...] [--resume "llm_response"]
"""

import sys
import json
import argparse
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(str(Path(__file__).parent))

from llm_agent_interface import LLMAgentInterface


def main():
    parser = argparse.ArgumentParser(description='Merge GCC branches with LLM assistance')
    parser.add_argument('--branches', nargs='+', help='Branch names to merge (required if not resuming)')
    parser.add_argument('--resume', type=str, help='LLM response to resume with')
    args = parser.parse_args()
    
    # Initialize interface (automatically loads persisted state)
    agent = LLMAgentInterface()
    
    if args.resume:
        # Resume with LLM response
        print("Resuming merge with LLM response...")
        result = agent.resume(args.resume)
        print("Resume result:")
        print(json.dumps(result, indent=2))
        return result
    
    if not args.branches:
        print("Error: --branches is required when not resuming")
        return
    
    # Execute merge command (will pause for LLM input)
    print(f"Merging branches: {', '.join(args.branches)}...")
    result = agent.merge_branch(args.branches)
    
    if result.get("status") == "paused_for_llm_input":
        print("PAUSE: LLM input needed for merge summary")
        print("Prompt for LLM:")
        print(json.dumps(result["prompt"], indent=2))
        print(f"\nAfter generating your response, call:")
        print(f"python merge.py --resume 'YOUR_MERGE_SUMMARY_TEXT'")
        print(f"Session ID: {result['session_id']}")
        return result
    else:
        print("Merge completed:")
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    main()