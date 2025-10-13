#!/usr/bin/env python3
"""
Initialize GCC project script for LLM agents.

Usage: python init.py [--project "project description"] [--resume "llm_response"]
"""

import sys
import json
import argparse
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(str(Path(__file__).parent))

from llm_agent_interface import LLMAgentInterface


def main():
    parser = argparse.ArgumentParser(description='Initialize GCC project with LLM assistance')
    parser.add_argument('--project', type=str, help='Optional project description')
    parser.add_argument('--resume', type=str, help='LLM response to resume with')
    args = parser.parse_args()
    
    if args.resume:
        # Initialize interface for resume
        agent = LLMAgentInterface()
        # Resume with LLM response
        print("Resuming project initialization with LLM response...")
        result = agent.resume(args.resume)
        print("Initialization completed:")
        print(json.dumps(result, indent=2))
        return result
    
    # Check if already initialized before creating agent
    gcc_path = Path(".") / ".GCC"
    if gcc_path.exists() and (gcc_path / "main.md").exists():
        print("GCC project already initialized. Use other commands to manage the project.")
        return {"status": "already_initialized"}
    
    # Initialize interface (will create .GCC structure)
    agent = LLMAgentInterface()
    
    # Execute initialization (will pause for LLM input)
    print("Initializing new GCC project...")
    result = agent.initialize_project(args.project)
    
    if result.get("status") == "paused_for_llm_input":
        print("PAUSE: LLM input needed for project initialization")
        print("Prompt for LLM:")
        print(json.dumps(result["prompt"], indent=2))
        print(f"\nAfter generating your response, call:")
        print(f"python init.py --resume 'YOUR_COMPLETE_MAIN_MD_CONTENT'")
        print(f"Session ID: {result['session_id']}")
        return result
    else:
        print("Project initialization completed:")
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    main()