#!/usr/bin/env python3
"""
Branch creation script for LLM agents.

Usage: python branch.py --name <branch_name> [--goal "description"] [--resume "llm_response"]
"""

import sys
import json
import argparse
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(str(Path(__file__).parent))

from llm_agent_interface import LLMAgentInterface


def main():
    parser = argparse.ArgumentParser(description='Create GCC branch with LLM assistance')
    parser.add_argument('--name', type=str, help='Branch name (required if not resuming)')
    parser.add_argument('--goal', type=str, help='Optional branch goal description')
    parser.add_argument('--resume', type=str, help='LLM response to resume with')
    args = parser.parse_args()
    
    # Initialize interface (automatically loads persisted state)
    agent = LLMAgentInterface()
    
    if args.resume:
        # Resume with LLM response
        print("Resuming branch creation with LLM response...")
        result = agent.resume(args.resume)
        print("Resume result:")
        print(json.dumps(result, indent=2))
        return result
    
    if not args.name:
        print("Error: --name is required when not resuming")
        return
    
    # Execute branch creation (will pause for LLM input)
    print(f"Creating branch '{args.name}'...")
    result = agent.create_branch(args.name, args.goal)
    
    if result.get("status") == "paused_for_llm_input":
        print("PAUSE: LLM input needed for branch purpose")
        print("Prompt for LLM:")
        print(json.dumps(result["prompt"], indent=2))
        print(f"\nAfter generating your response, call:")
        print(f"python branch.py --resume 'YOUR_BRANCH_PURPOSE_TEXT'")
        print(f"Session ID: {result['session_id']}")
        return result
    else:
        print("Branch creation completed:")
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    main()