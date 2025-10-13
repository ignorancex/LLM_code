"""
LLM Agent Interface for Git-Context-Controller.

This module provides a specialized interface designed for LLM agents to interact
with GCC system programmatically, including pause/resume functionality and
structured command execution patterns based on the GCC workflow specification.
"""

import json
import sys
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path

from core import GitContextController, GCCConfig
from commands import GCCCommandInterface


class LLMAgentInterface:
    """
    Specialized interface for LLM agents to interact with GCC system.
    
    Provides pause/resume functionality, structured prompting, and
    programmatic command execution designed for agentic workflows.
    Based on the GCC workflow specification for optimal LLM interaction.
    """
    
    def __init__(self, project_root: Union[str, Path] = ".", config: Optional[GCCConfig] = None):
        """Initialize LLM Agent Interface."""
        self.controller = GitContextController(project_root, config)
        self.commands = GCCCommandInterface(self.controller)
        self.session_id = f"llm_session_{int(time.time())}"
        self.paused_state = None
        self.interaction_count = 0
        self.state_file = self.controller.gcc_path / "llm_agent_state.json"
        
        # Load existing state if available
        self._load_state()
        
    def commit(self, progress_description: str = None) -> Dict[str, Any]:
        """
        Execute COMMIT command with LLM-guided summary generation.
        
        Pauses to ask LLM to generate commit.md content based on last commit
        and current progress.
        """
        self.interaction_count += 1
        
        # Get last commit content for context
        last_commit_content = self._get_last_commit_content()
        
        return self._create_commit_prompt(last_commit_content, progress_description)
    
    def create_branch(self, name: str, goal: str = None) -> Dict[str, Any]:
        """
        Execute BRANCH command with LLM-guided branch planning.
        
        Pauses to ask LLM to generate simple branch purpose.
        """
        self.interaction_count += 1
        
        return self._create_branch_prompt(name, goal)
    
    def merge_branch(self, branch_names: List[str]) -> Dict[str, Any]:
        """
        Execute MERGE command with LLM-guided synthesis.
        
        Pauses to ask LLM to generate merge summary.
        """
        self.interaction_count += 1
        
        # Get branch summaries for context
        branch_summaries = self._get_branch_summaries(branch_names)
        
        return self._create_merge_prompt(branch_summaries, branch_names)
    
    def show_context(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute CONTEXT command by reading stored context files.
        
        Does NOT call LLM - directly reads and returns context from files
        as specified in the GCC workflow.
        """
        self.interaction_count += 1
        
        if not options:
            # Default git status-style snapshot
            return self._get_git_status_context()
        
        # Handle specific context requests
        if options.get("branch"):
            return self._get_branch_context(options["branch"])
        elif options.get("commit"):
            return self._get_commit_context(options["commit"])
        elif options.get("log"):
            return self._get_log_context(options.get("lines", 20))
        elif options.get("metadata"):
            return self._get_metadata_context(options["metadata"])
        else:
            return self._get_git_status_context()
    
    def update_main_roadmap(self, merge_summary: str = None) -> Dict[str, Any]:
        """
        Update main.md roadmap by prompting LLM.
        
        Called after commit (optionally), branch (definitely), or merge (definitely).
        """
        self.interaction_count += 1
        
        current_main = self._get_current_main_content()
        current_commit = self._get_last_commit_content()
        
        return self._create_main_update_prompt(current_main, current_commit)
    
    def initialize_project(self, project_description: str = None) -> Dict[str, Any]:
        """
        Initialize new GCC project with LLM-guided main.md creation.
        
        Pauses to ask LLM to create initial project roadmap and goals.
        """
        self.interaction_count += 1
        
        # Initialize GCC structure if needed
        if not self.controller.gcc_path.exists():
            self.controller._initialize_gcc()
        
        return self._create_init_prompt(project_description)
    
    def resume(self, llm_response: str) -> Dict[str, Any]:
        """Resume execution with LLM's response."""
        if not self.paused_state:
            return self._format_error_response("No paused state to resume from")
        
        command = self.paused_state["command"]
        
        try:
            if command == "commit":
                result = self._complete_commit(llm_response)
            elif command == "create_branch":
                result = self._complete_branch_creation(llm_response)
            elif command == "merge_branch":
                result = self._complete_merge(llm_response)
            elif command == "update_main_roadmap":
                result = self._complete_main_update(llm_response)
            elif command == "initialize_project":
                result = self._complete_project_initialization(llm_response)
            else:
                result = self._format_error_response(f"Unknown paused command: {command}")
            
            # Clear paused state both in memory and on disk
            self._clear_state()
            
            return self._format_success_response("resume", result)
            
        except Exception as e:
            self._clear_state()
            return self._format_error_response(f"Resume failed: {str(e)}")
    
    # Context reading methods (no LLM involved)
    
    def _get_git_status_context(self) -> Dict[str, Any]:
        """Get git status-style snapshot."""
        main_content = self._get_current_main_content()
        available_branches = self.controller.list_branches()
        
        return {
            "action": "show_context",
            "project_purpose": main_content[:500] + "..." if len(main_content) > 500 else main_content,
            "available_branches": available_branches,
            "current_branch": self.controller.current_branch,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_branch_context(self, branch_name: str) -> Dict[str, Any]:
        """Get specific branch context."""
        branch_path = self.controller.gcc_path / "branches" / branch_name
        
        if not branch_path.exists():
            return {"error": f"Branch {branch_name} does not exist"}
        
        commit_content = ""
        if (branch_path / "commit.md").exists():
            commit_content = (branch_path / "commit.md").read_text()
        
        # Get last 10 commits summary
        commits = self._extract_recent_commits(commit_content, 10)
        
        return {
            "action": "branch_context",
            "branch": branch_name,
            "purpose_and_progress": commit_content[:1000] + "..." if len(commit_content) > 1000 else commit_content,
            "recent_commits": commits
        }
    
    def _get_commit_context(self, commit_hash: str) -> Dict[str, Any]:
        """Get specific commit context."""
        current_branch = self.controller.current_branch
        commit_file = self.controller.gcc_path / "branches" / current_branch / "commit.md"
        
        if commit_file.exists():
            content = commit_file.read_text()
            return {
                "action": "commit_context",
                "commit_hash": commit_hash,
                "full_commit_content": content
            }
        
        return {"error": "No commit content found"}
    
    def _get_log_context(self, lines: int = 20) -> Dict[str, Any]:
        """Get execution log context."""
        current_branch = self.controller.current_branch
        log_file = self.controller.gcc_path / "branches" / current_branch / "log.md"
        
        if log_file.exists():
            content = log_file.read_text()
            log_lines = content.split('\n')
            recent_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines
            
            return {
                "action": "log_context",
                "branch": current_branch,
                "recent_log": '\n'.join(recent_lines),
                "lines_returned": len(recent_lines)
            }
        
        return {"error": "No log content found"}
    
    def _get_metadata_context(self, segment: str) -> Dict[str, Any]:
        """Get metadata context for specific segment."""
        current_branch = self.controller.current_branch
        metadata_file = self.controller.gcc_path / "branches" / current_branch / "metadata.yaml"
        
        if metadata_file.exists():
            import yaml
            with open(metadata_file) as f:
                metadata = yaml.safe_load(f)
            
            if segment in metadata:
                return {
                    "action": "metadata_context",
                    "segment": segment,
                    "content": metadata[segment]
                }
            else:
                return {
                    "action": "metadata_context",
                    "available_segments": list(metadata.keys()),
                    "error": f"Segment {segment} not found"
                }
        
        return {"error": "No metadata found"}
    
    # Helper methods for getting context data
    
    def _get_last_commit_content(self) -> str:
        """Get content of last commit for context."""
        current_branch = self.controller.current_branch
        commit_file = self.controller.gcc_path / "branches" / current_branch / "commit.md"
        
        if commit_file.exists():
            return commit_file.read_text()
        return ""
    
    def _get_branch_summaries(self, branch_names: List[str]) -> Dict[str, str]:
        """Get commit.md content for specified branches."""
        summaries = {}
        for branch in branch_names:
            branch_path = self.controller.gcc_path / "branches" / branch
            commit_file = branch_path / "commit.md"
            if commit_file.exists():
                summaries[branch] = commit_file.read_text()
            else:
                summaries[branch] = f"No commit history for branch {branch}"
        return summaries
    
    def _get_current_main_content(self) -> str:
        """Get current main.md content."""
        main_file = self.controller.gcc_path / "main.md"
        if main_file.exists():
            return main_file.read_text()
        return ""
    
    def _extract_recent_commits(self, commit_content: str, count: int) -> List[str]:
        """Extract recent commit messages from commit.md content."""
        lines = commit_content.split('\n')
        commits = []
        
        for line in lines:
            if line.startswith('## Commit:'):
                timestamp = line.replace('## Commit:', '').strip()
                commits.append(f"Commit {timestamp}")
                if len(commits) >= count:
                    break
        
        return commits[-count:] if len(commits) > count else commits
    
    # Prompt creation methods
    
    def _create_commit_prompt(self, last_commit_content: str, progress_description: str = None) -> Dict[str, Any]:
        """Create LLM prompt for commit.md generation."""
        self.paused_state = {
            "command": "commit",
            "last_commit_content": last_commit_content,
            "progress_description": progress_description,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "interaction_count": self.interaction_count
        }
        
        prompt = {
            "task": "Generate complete commit.md content for GCC system",
            "instructions": [
                "Append a new commit entry to the existing commit.md file.",
                "Follow the GCC commit.md format with three required blocks:",
                "1. Branch Purpose - reiteration of overall project goal and branch rationale",
                "2. Previous Progress Summary - coarse-grained summary combining last commit with recent work", 
                "3. This Commit's Contribution - detailed narrative of current commit achievements",
                "Return the complete updated commit.md content."
            ],
            "context_provided": {
                "existing_commit_content": last_commit_content[:2000] + "..." if len(last_commit_content) > 2000 else last_commit_content,
                "current_branch": self.controller.current_branch,
                "progress_hint": progress_description or "Analyze your recent work and progress"
            },
            "output_format": "Complete commit.md file content as plain text (not JSON)"
        }
        
        # Save state before pausing
        self._save_state()
        
        return {
            "status": "paused_for_llm_input",
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "command": "commit",
            "prompt": prompt,
            "instruction": "Generate the complete updated commit.md content.",
            "resume_instruction": "Call resume() with the complete commit.md content"
        }
    
    def _create_branch_prompt(self, name: str, goal: str = None) -> Dict[str, Any]:
        """Create LLM prompt for simple branch purpose generation."""
        self.paused_state = {
            "command": "create_branch",
            "branch_name": name,
            "suggested_goal": goal,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "interaction_count": self.interaction_count
        }
        
        prompt = {
            "task": f"Generate simple branch purpose for new branch '{name}'",
            "instructions": [
                "Generate a brief, clear explanation of this branch's purpose.",
                "Keep it concise - just explain why this branch represents a different direction.",
                "This will be used in the branch's commit.md file."
            ],
            "context_provided": {
                "branch_name": name,
                "suggested_goal": goal or "Not specified",
                "current_branch": self.controller.current_branch
            },
            "output_format": "Brief branch purpose as plain text (not JSON)"
        }
        
        # Save state before pausing
        self._save_state()
        
        return {
            "status": "paused_for_llm_input",
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "command": "create_branch",
            "prompt": prompt,
            "instruction": "Generate a brief branch purpose explanation.",
            "resume_instruction": "Call resume() with the branch purpose text"
        }
    
    def _create_merge_prompt(self, branch_summaries: Dict[str, str], branch_names: List[str]) -> Dict[str, Any]:
        """Create LLM prompt for merge summary generation."""
        self.paused_state = {
            "command": "merge_branch",
            "branch_summaries": branch_summaries,
            "branch_names": branch_names,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "interaction_count": self.interaction_count
        }
        
        # Truncate summaries for prompt
        truncated_summaries = {}
        for branch, content in branch_summaries.items():
            truncated_summaries[branch] = content[:1000] + "..." if len(content) > 1000 else content
        
        prompt = {
            "task": f"Generate merge summary for branches: {', '.join(branch_names)}",
            "instructions": [
                "Analyze the progress from each branch to be merged.",
                "Generate a concise merge summary explaining what is being unified.",
                "Focus on the synthesis and combined value of the merged branches."
            ],
            "context_provided": {
                "branches_to_merge": branch_names,
                "branch_summaries": truncated_summaries,
                "target_branch": self.controller.current_branch
            },
            "output_format": "Merge summary as plain text (not JSON)"
        }
        
        # Save state before pausing
        self._save_state()
        
        return {
            "status": "paused_for_llm_input",
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "command": "merge_branch",
            "prompt": prompt,
            "instruction": "Generate the merge summary.",
            "resume_instruction": "Call resume() with the merge summary text"
        }
    
    def _create_main_update_prompt(self, current_main: str, current_commit: str) -> Dict[str, Any]:
        """Create LLM prompt for main.md update with previous main.md and current commit.md."""
        self.paused_state = {
            "command": "update_main_roadmap",
            "current_main": current_main,
            "current_commit": current_commit,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "interaction_count": self.interaction_count
        }
        
        prompt = {
            "task": "Update main.md roadmap based on recent commit progress",
            "instructions": [
                "Update the main project roadmap to reflect recent progress from commit.md.",
                "The commit.md contains the latest progress including any merge summaries or branch purposes.",
                "Update milestones, goals, and planning state as appropriate.",
                "Return the complete updated main.md content."
            ],
            "context_provided": {
                "current_main_content": current_main[:1500] + "..." if len(current_main) > 1500 else current_main,
                "current_commit_content": current_commit[:1500] + "..." if len(current_commit) > 1500 else current_commit,
                "current_branch": self.controller.current_branch
            },
            "output_format": "Complete main.md file content as plain text (not JSON)"
        }
        
        # Save state before pausing
        self._save_state()
        
        return {
            "status": "paused_for_llm_input",
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "command": "update_main_roadmap",
            "prompt": prompt,
            "instruction": "Generate the complete updated main.md content.",
            "resume_instruction": "Call resume() with the complete main.md content"
        }
    
    def _create_init_prompt(self, project_description: str = None) -> Dict[str, Any]:
        """Create LLM prompt for project initialization."""
        self.paused_state = {
            "command": "initialize_project",
            "project_description": project_description,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "interaction_count": self.interaction_count
        }
        
        prompt = {
            "task": "Generate project introduction with description, goals, and plan",
            "instructions": [
                "Create a comprehensive project introduction including:",
                "- Project description and purpose",
                "- Main goals and objectives", 
                "- High-level plan and approach",
                "- Key milestones or phases",
                "Write it professionally and clearly."
            ],
            "context_provided": {
                "project_hint": project_description or "No specific description provided",
                "project_location": str(self.controller.project_root)
            },
            "output_format": "Project introduction as plain text"
        }
        
        # Save state before pausing
        self._save_state()
        
        return {
            "status": "paused_for_llm_input",
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "command": "initialize_project",
            "prompt": prompt,
            "instruction": "Generate the project introduction with description, goals, and plan.",
            "resume_instruction": "Call resume() with your project introduction"
        }
    
    # Completion methods - one-shot updates
    
    def _complete_commit(self, llm_response: str) -> Dict[str, Any]:
        """Complete commit by updating commit.md with LLM response."""
        current_branch = self.controller.current_branch
        commit_file = self.controller.gcc_path / "branches" / current_branch / "commit.md"
        
        # Write LLM response directly to commit.md
        commit_file.write_text(llm_response.strip())
        
        # Also append OTA log if any
        if self.controller.current_ota_log:
            self.controller._append_ota_to_log()
            self.controller.current_ota_log = []
        
        # Create git commit if enabled
        if self.controller.config.git_integration:
            try:
                import subprocess
                subprocess.run(["git", "add", str(self.controller.gcc_path)], 
                             cwd=self.controller.project_root, check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", f"GCC: Updated commit for {current_branch}"], 
                             cwd=self.controller.project_root, check=True, capture_output=True)
            except:
                pass
        
        return {
            "action": "commit_completed",
            "branch": current_branch,
            "commit_file_updated": str(commit_file),
            "timestamp": datetime.now().isoformat()
        }
    
    def _complete_branch_creation(self, llm_response: str) -> Dict[str, Any]:
        """Complete branch creation with LLM-generated purpose."""
        branch_name = self.paused_state["branch_name"]
        
        # Create branch structure
        self.controller._create_branch_structure(branch_name)
        
        # Write LLM-generated purpose to commit.md
        branch_path = self.controller.gcc_path / "branches" / branch_name
        commit_file = branch_path / "commit.md"
        
        branch_content = f"""# Branch: {branch_name}

## Branch Purpose
{llm_response.strip()}

**Created from:** {self.controller.current_branch}
**Created at:** {datetime.now().isoformat()}

## Commit History

"""
        commit_file.write_text(branch_content)
        
        # Update metadata
        self.controller._update_branch_metadata({
            "branch_info": {
                "name": branch_name,
                "created": datetime.now().isoformat(),
                "purpose": llm_response.strip(),
                "parent_branch": self.controller.current_branch
            }
        }, branch_name=branch_name)
        
        # Switch to new branch
        previous_branch = self.controller.current_branch
        self.controller.current_branch = branch_name
        
        # Automatically trigger main.md update after branch creation
        main_update_result = self.update_main_roadmap()
        
        return {
            "action": "branch_created",
            "branch_name": branch_name,
            "purpose": llm_response.strip(),
            "previous_branch": previous_branch,
            "timestamp": datetime.now().isoformat(),
            "main_update_needed": main_update_result
        }
    
    def _complete_merge(self, llm_response: str) -> Dict[str, Any]:
        """Complete merge with LLM-generated summary."""
        branch_names = self.paused_state["branch_names"]
        merge_summary = llm_response.strip()
        
        # Perform the actual merge using the controller
        merge_result = self.controller.merge(branch_names, merge_summary)
        
        # Update current branch's commit.md with merge summary as this-commit-contribution
        current_branch = self.controller.current_branch
        commit_file = self.controller.gcc_path / "branches" / current_branch / "commit.md"
        
        # Append merge entry to commit.md
        merge_entry = f"""
## Commit: {datetime.now().isoformat()}
**Summary:** Merged branches: {', '.join(branch_names)}
**Branch:** {current_branch}

### This Commit's Contribution
{merge_summary}

---
"""
        
        if commit_file.exists():
            current_content = commit_file.read_text()
            commit_file.write_text(current_content + merge_entry)
        else:
            commit_file.write_text(merge_entry)
        
        # Automatically trigger main.md update after merge
        main_update_result = self.update_main_roadmap()
        
        return {
            "action": "merge_completed",
            "merged_branches": branch_names,
            "merge_summary": merge_summary,
            "target_branch": current_branch,
            "timestamp": datetime.now().isoformat(),
            "main_update_needed": main_update_result
        }
    
    def _complete_main_update(self, llm_response: str) -> Dict[str, Any]:
        """Complete main.md update with LLM response."""
        main_file = self.controller.gcc_path / "main.md"
        
        # Write LLM response directly to main.md
        main_file.write_text(llm_response.strip())
        
        return {
            "action": "main_roadmap_updated",
            "main_file": str(main_file),
            "timestamp": datetime.now().isoformat()
        }
    
    def _complete_project_initialization(self, llm_response: str) -> Dict[str, Any]:
        """Complete project initialization by creating main.md with LLM content."""
        # Ensure GCC structure exists
        if not self.controller.gcc_path.exists():
            self.controller._initialize_gcc()
        
        # Create main.md with LLM-generated content
        main_file = self.controller.gcc_path / "main.md"
        main_file.write_text(llm_response.strip())
        
        return {
            "action": "project_initialized",
            "main_file": str(main_file),
            "gcc_directory": str(self.controller.gcc_path),
            "timestamp": datetime.now().isoformat()
        }
    
    # State persistence methods
    
    def _save_state(self):
        """Save current state to disk."""
        state = {
            "session_id": self.session_id,
            "paused_state": self.paused_state,
            "interaction_count": self.interaction_count,
            "current_branch": self.controller.current_branch
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from disk if available."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.session_id = state.get("session_id", self.session_id)
                self.paused_state = state.get("paused_state")
                self.interaction_count = state.get("interaction_count", 0)
                
                # Restore branch state
                if state.get("current_branch"):
                    self.controller.current_branch = state["current_branch"]
                    
            except (json.JSONDecodeError, KeyError):
                # If state file is corrupted, start fresh
                pass
    
    def _clear_state(self):
        """Clear persisted state."""
        if self.state_file.exists():
            self.state_file.unlink()
        self.paused_state = None
    
    # Utility methods
    
    def _format_success_response(self, command: str, result: Any) -> Dict[str, Any]:
        """Format successful command response."""
        return {
            "status": "success",
            "command": command,
            "result": result,
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """Format error response."""
        return {
            "status": "error",
            "error": error_message,
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current interface status."""
        return {
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "is_paused": self.paused_state is not None,
            "paused_command": self.paused_state["command"] if self.paused_state else None,
            "current_branch": self.controller.current_branch,
            "available_branches": self.controller.list_branches(),
            "gcc_path": str(self.controller.gcc_path),
            "timestamp": datetime.now().isoformat()
        }