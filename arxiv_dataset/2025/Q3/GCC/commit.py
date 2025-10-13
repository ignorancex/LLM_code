#!/usr/bin/env python3
"""
Commit script for LLM agents.

Usage: python commit.py [--progress "description"] [--resume "llm_response"]
"""

import sys
import json
import argparse
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(str(Path(__file__).parent))

from llm_agent_interface import LLMAgentInterface


def main():
    parser = argparse.ArgumentParser(description='Execute GCC commit with LLM assistance')
    parser.add_argument('--progress', type=str, help='Optional progress description')
    parser.add_argument('--resume', type=str, help='LLM response to resume with')
    args = parser.parse_args()
    
    # Initialize interface (automatically loads persisted state)
    agent = LLMAgentInterface()
    
    if args.resume:
        # Resume with LLM response
        print("Resuming commit with LLM response...")
        result = agent.resume(args.resume)
        print("Resume result:")
        print(json.dumps(result, indent=2))
        return result
    
    # Execute commit command (will pause for LLM input)
    print("Executing commit command...")
    result = agent.commit(args.progress)
    
    if result.get("status") == "paused_for_llm_input":
        print("PAUSE: LLM input needed")
        print("Prompt for LLM:")
        print(json.dumps(result["prompt"], indent=2))
        print(f"\nAfter generating your response, call:")
        print(f"python commit.py --resume 'YOUR_COMPLETE_COMMIT_MD_CONTENT'")
        print(f"Session ID: {result['session_id']}")
        return result
    else:
        print("Commit completed:")
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    main()