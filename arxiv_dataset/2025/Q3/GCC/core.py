"""
Core Git-Context-Controller implementation.

This module provides the main GitContextController class that manages the structured
memory system using a Git-like approach to context versioning and branching.
"""

import os
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict


@dataclass
class GCCConfig:
    """Configuration for Git-Context-Controller."""
    gcc_dir: str = ".GCC"
    default_branch: str = "main"
    git_integration: bool = True
    auto_compress_threshold: int = 50  # Compress after N commits
    max_log_entries: int = 1000       # Max OTA entries per branch


@dataclass
class OTAStep:
    """Represents a single Observation-Thought-Action step."""
    timestamp: str
    observation: str
    thought: str
    action: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CommitInfo:
    """Information about a commit."""
    timestamp: str
    summary: str
    branch: str
    ota_count: int
    metadata: Optional[Dict[str, Any]] = None


class GitContextController:
    """
    Main controller for Git-Controlled Context system.
    
    This class manages a structured memory system organized like a Git repository,
    with branches for parallel exploration and commits for milestone tracking.
    """
    
    def __init__(self, project_root: Union[str, Path] = ".", config: Optional[GCCConfig] = None):
        """
        Initialize the Git-Context-Controller.
        
        Args:
            project_root: Root directory for the project
            config: Configuration object, uses default if None
        """
        self.project_root = Path(project_root).resolve()
        self.config = config or GCCConfig()
        self.gcc_path = self.project_root / self.config.gcc_dir
        self.current_branch = self.config.default_branch
        self.current_ota_log: List[OTAStep] = []
        
        # Initialize GCC directory structure
        self._initialize_gcc()
    
    def _initialize_gcc(self):
        """Initialize the .GCC directory structure."""
        self.gcc_path.mkdir(exist_ok=True)
        
        # Create main.md if it doesn't exist
        main_md = self.gcc_path / "main.md"
        if not main_md.exists():
            main_md.write_text(self._get_default_main_content())
        
        # Create branches directory
        branches_dir = self.gcc_path / "branches"
        branches_dir.mkdir(exist_ok=True)
        
        # Create default branch
        self._create_branch_structure(self.config.default_branch)
    
    def _get_default_main_content(self) -> str:
        """Get default content for main.md."""
        return """# Project Roadmap

## High-level Intent
This project uses Git-Controlled Context (GCC) for structured memory management.

## Milestones
- [ ] Project initialization
- [ ] Core functionality implementation
- [ ] Testing and validation
- [ ] Documentation and deployment

## Shared Planning State
GCC system initialized and ready for structured reasoning.

Last updated: {timestamp}
""".format(timestamp=datetime.now().isoformat())
    
    def _create_branch_structure(self, branch_name: str):
        """Create directory structure for a branch."""
        branch_path = self.gcc_path / "branches" / branch_name
        branch_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize branch files
        commit_md = branch_path / "commit.md"
        if not commit_md.exists():
            commit_md.write_text(f"# Branch: {branch_name}\n\n## Commit History\n\n")
        
        log_md = branch_path / "log.md"
        if not log_md.exists():
            log_md.write_text("# OTA Execution Log\n\n")
        
        metadata_yaml = branch_path / "metadata.yaml"
        if not metadata_yaml.exists():
            initial_metadata = {
                "branch_info": {
                    "name": branch_name,
                    "created": datetime.now().isoformat(),
                    "purpose": "Main development branch" if branch_name == self.config.default_branch else "Feature branch"
                },
                "file_structure": {},
                "env_config": {},
                "dependencies": [],
                "custom_data": {}
            }
            with open(metadata_yaml, 'w') as f:
                yaml.dump(initial_metadata, f, default_flow_style=False)
    
    def commit(self, summary: str, metadata: Optional[Dict[str, Any]] = None) -> CommitInfo:
        """
        Create a new checkpoint/commit.
        
        Args:
            summary: Short summary of what was accomplished
            metadata: Optional metadata to store with commit
            
        Returns:
            CommitInfo object with commit details
        """
        timestamp = datetime.now().isoformat()
        branch_path = self.gcc_path / "branches" / self.current_branch
        
        # Create commit info
        commit_info = CommitInfo(
            timestamp=timestamp,
            summary=summary,
            branch=self.current_branch,
            ota_count=len(self.current_ota_log),
            metadata=metadata
        )
        
        # Update commit.md with new entry
        self._update_commit_md(commit_info)
        
        # Append current OTA cycle to log.md
        self._append_ota_to_log()
        
        # Update metadata if provided
        if metadata:
            self._update_branch_metadata(metadata)
        
        # Clear current OTA log for next cycle
        self.current_ota_log = []
        
        # Optional: Create git commit if enabled
        if self.config.git_integration:
            self._create_git_commit(summary)
        
        return commit_info
    
    def create_branch(self, name: str, goal: str, copy_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new branch for alternative exploration.
        
        Args:
            name: Branch name
            goal: Purpose/goal of the branch
            copy_from: Optional branch to copy from (defaults to current branch)
            
        Returns:
            Dictionary with branch creation details
        """
        if copy_from is None:
            copy_from = self.current_branch
        
        # Create branch structure
        self._create_branch_structure(name)
        
        # Copy content from source branch if it exists and is different
        if copy_from != name and copy_from != self.config.default_branch:
            self._copy_branch_content(copy_from, name)
        
        # Update branch purpose in commit.md
        branch_path = self.gcc_path / "branches" / name
        commit_md = branch_path / "commit.md"
        
        branch_purpose = f"""# Branch: {name}

## Branch Purpose
**Goal:** {goal}
**Created from:** {copy_from}
**Created at:** {datetime.now().isoformat()}

## Previous Progress Summary
Branch created from {copy_from} to: {goal}

## Commit History

"""
        commit_md.write_text(branch_purpose)
        
        # Update metadata
        self._update_branch_metadata({
            "branch_info": {
                "name": name,
                "created": datetime.now().isoformat(),
                "purpose": goal,
                "parent_branch": copy_from
            }
        }, branch_name=name)
        
        # Switch to new branch
        previous_branch = self.current_branch
        self.current_branch = name
        
        return {
            "action": "create_branch",
            "name": name,
            "goal": goal,
            "parent_branch": copy_from,
            "previous_branch": previous_branch,
            "timestamp": datetime.now().isoformat()
        }
    
    def merge(self, branches: List[str], summary: str, target_branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Consolidate progress from multiple branches.
        
        Args:
            branches: List of branch names to merge
            summary: Summary of merged conclusions
            target_branch: Target branch (defaults to current branch)
            
        Returns:
            Dictionary with merge details
        """
        if target_branch is None:
            target_branch = self.current_branch
        
        target_branch_path = self.gcc_path / "branches" / target_branch
        
        # Collect content from all branches
        merged_commits = []
        merged_logs = []
        merged_metadata = {}
        
        for branch in branches:
            branch_path = self.gcc_path / "branches" / branch
            if not branch_path.exists():
                continue
                
            # Collect commit history
            commit_content = (branch_path / "commit.md").read_text()
            merged_commits.append(f"## Merged from Branch: {branch}\n{commit_content}")
            
            # Collect logs with origin tags
            log_content = (branch_path / "log.md").read_text()
            merged_logs.append(f"== Branch {branch} ==\n{log_content}")
            
            # Collect metadata
            with open(branch_path / "metadata.yaml") as f:
                branch_metadata = yaml.safe_load(f)
                merged_metadata[branch] = branch_metadata
        
        # Update target branch with merged content
        timestamp = datetime.now().isoformat()
        
        # Update commit.md with merge summary
        merge_entry = f"""
## Merge: {timestamp}
**Summary:** {summary}
**Merged branches:** {', '.join(branches)}
**Target branch:** {target_branch}

### Merged Content Summary
{chr(10).join(merged_commits)}

---
"""
        
        with open(target_branch_path / "commit.md", 'a') as f:
            f.write(merge_entry)
        
        # Update log.md with merged logs
        with open(target_branch_path / "log.md", 'a') as f:
            f.write(f"\n## Merge: {timestamp}\n")
            f.write('\n'.join(merged_logs))
        
        # Update metadata with merged information
        target_metadata_path = target_branch_path / "metadata.yaml"
        with open(target_metadata_path) as f:
            target_metadata = yaml.safe_load(f)
        
        target_metadata["merged_branches"] = merged_metadata
        target_metadata["last_merge"] = {
            "timestamp": timestamp,
            "branches": branches,
            "summary": summary
        }
        
        with open(target_metadata_path, 'w') as f:
            yaml.dump(target_metadata, f, default_flow_style=False)
        
        # Switch to target branch
        self.current_branch = target_branch
        
        return {
            "action": "merge",
            "branches": branches,
            "target_branch": target_branch,
            "summary": summary,
            "timestamp": timestamp
        }
    
    def show_context(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve project history and context.
        
        Args:
            options: Optional context retrieval options
            
        Returns:
            Dictionary with context information
        """
        # Read main.md for global roadmap
        main_content = (self.gcc_path / "main.md").read_text()
        
        # List available branches
        branches_dir = self.gcc_path / "branches"
        available_branches = [d.name for d in branches_dir.iterdir() if d.is_dir()]
        
        # Get current branch status
        current_branch_path = self.gcc_path / "branches" / self.current_branch
        commit_content = (current_branch_path / "commit.md").read_text()
        
        # Basic context
        context = {
            "action": "show_context",
            "timestamp": datetime.now().isoformat(),
            "main_roadmap": main_content,
            "available_branches": available_branches,
            "current_branch": self.current_branch,
            "current_commit_history": commit_content
        }
        
        # Add specific context based on options
        if options:
            if options.get("include_metadata"):
                context["current_metadata"] = self.get_branch_metadata(self.current_branch)
            
            if options.get("include_log_lines"):
                lines = options.get("log_lines", 20)
                context["recent_log"] = self.get_branch_log(self.current_branch, lines)
            
            if options.get("branch_details"):
                context["branch_details"] = {}
                for branch in available_branches:
                    context["branch_details"][branch] = {
                        "metadata": self.get_branch_metadata(branch),
                        "recent_commits": self._get_recent_commits(branch, 5)
                    }
        
        return context
    
    def add_ota_step(self, observation: str, thought: str, action: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an OTA (Observation-Thought-Action) step to current reasoning cycle.
        
        Args:
            observation: What was observed
            thought: What was thought/inferred  
            action: What action was taken
            metadata: Optional metadata for this step
        """
        ota_step = OTAStep(
            timestamp=datetime.now().isoformat(),
            observation=observation,
            thought=thought,
            action=action,
            metadata=metadata
        )
        self.current_ota_log.append(ota_step)
        
        # Auto-compress if threshold reached
        if len(self.current_ota_log) >= self.config.max_log_entries:
            self._compress_ota_log()
    
    def get_branch_context(self, branch: str, include_log: bool = False, log_lines: int = 20) -> Dict[str, Any]:
        """Get detailed context for a specific branch."""
        branch_path = self.gcc_path / "branches" / branch
        
        if not branch_path.exists():
            return {"error": f"Branch {branch} does not exist"}
        
        context = {
            "branch": branch,
            "commit_history": (branch_path / "commit.md").read_text(),
            "metadata": self.get_branch_metadata(branch)
        }
        
        if include_log:
            context["execution_log"] = self.get_branch_log(branch, log_lines)
        
        return context
    
    def get_branch_log(self, branch: str, lines: Optional[int] = None) -> str:
        """Get execution log for a specific branch."""
        branch_path = self.gcc_path / "branches" / branch
        log_md = branch_path / "log.md"
        
        if not log_md.exists():
            return ""
        
        content = log_md.read_text()
        if lines:
            log_lines = content.split('\n')
            recent_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines
            return '\n'.join(recent_lines)
        return content
    
    def get_branch_metadata(self, branch: str) -> Dict[str, Any]:
        """Get metadata for a specific branch."""
        branch_path = self.gcc_path / "branches" / branch
        metadata_yaml = branch_path / "metadata.yaml"
        
        if metadata_yaml.exists():
            with open(metadata_yaml) as f:
                return yaml.safe_load(f)
        return {}
    
    def update_branch_metadata(self, metadata: Dict[str, Any], branch: Optional[str] = None):
        """Update metadata for a branch."""
        self._update_branch_metadata(metadata, branch or self.current_branch)
    
    def list_branches(self) -> List[str]:
        """List all available branches."""
        branches_dir = self.gcc_path / "branches"
        if branches_dir.exists():
            return [d.name for d in branches_dir.iterdir() if d.is_dir()]
        return []
    
    def switch_branch(self, branch: str) -> bool:
        """Switch to a different branch."""
        branch_path = self.gcc_path / "branches" / branch
        if branch_path.exists():
            self.current_branch = branch
            return True
        return False
    
    def delete_branch(self, branch: str, force: bool = False) -> bool:
        """Delete a branch."""
        if branch == self.config.default_branch and not force:
            return False
        
        branch_path = self.gcc_path / "branches" / branch
        if branch_path.exists():
            import shutil
            shutil.rmtree(branch_path)
            
            # Switch to default branch if we deleted current branch
            if self.current_branch == branch:
                self.current_branch = self.config.default_branch
            
            return True
        return False
    
    # Private helper methods
    
    def _update_commit_md(self, commit_info: CommitInfo):
        """Update commit.md with new commit entry."""
        branch_path = self.gcc_path / "branches" / commit_info.branch
        commit_md = branch_path / "commit.md"
        
        commit_entry = f"""
## Commit: {commit_info.timestamp}
**Summary:** {commit_info.summary}
**OTA Steps:** {commit_info.ota_count}
**Branch:** {commit_info.branch}

### This Commit's Contribution
{commit_info.summary}

"""
        
        if commit_info.metadata:
            commit_entry += f"**Metadata:** {json.dumps(commit_info.metadata, indent=2)}\n"
        
        commit_entry += "---\n"
        
        with open(commit_md, 'a') as f:
            f.write(commit_entry)
    
    def _append_ota_to_log(self):
        """Append current OTA cycle to log.md."""
        if not self.current_ota_log:
            return
        
        branch_path = self.gcc_path / "branches" / self.current_branch
        log_md = branch_path / "log.md"
        
        log_entry = f"\n## OTA Cycle: {datetime.now().isoformat()}\n\n"
        
        for i, ota in enumerate(self.current_ota_log, 1):
            log_entry += f"### Step {i} ({ota.timestamp})\n"
            log_entry += f"**OBSERVATION:** {ota.observation}\n\n"
            log_entry += f"**THOUGHT:** {ota.thought}\n\n"  
            log_entry += f"**ACTION:** {ota.action}\n\n"
            
            if ota.metadata:
                log_entry += f"**METADATA:** {json.dumps(ota.metadata, indent=2)}\n\n"
            
            log_entry += "---\n\n"
        
        with open(log_md, 'a') as f:
            f.write(log_entry)
    
    def _update_branch_metadata(self, metadata: Dict[str, Any], branch_name: Optional[str] = None):
        """Update metadata for a specific branch."""
        if branch_name is None:
            branch_name = self.current_branch
            
        branch_path = self.gcc_path / "branches" / branch_name
        metadata_path = branch_path / "metadata.yaml"
        
        existing_metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                existing_metadata = yaml.safe_load(f) or {}
        
        # Deep merge metadata
        self._deep_merge_dict(existing_metadata, metadata)
        existing_metadata["last_updated"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            yaml.dump(existing_metadata, f, default_flow_style=False)
    
    def _deep_merge_dict(self, target: Dict, source: Dict):
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value
    
    def _copy_branch_content(self, source_branch: str, target_branch: str):
        """Copy content from source branch to target branch."""
        source_path = self.gcc_path / "branches" / source_branch
        target_path = self.gcc_path / "branches" / target_branch
        
        if not source_path.exists():
            return
        
        # Copy metadata and recent log content
        try:
            source_metadata = self.get_branch_metadata(source_branch)
            self._update_branch_metadata(source_metadata, target_branch)
            
            # Copy recent log entries (last 10)
            source_log = self.get_branch_log(source_branch, 200)  # Get recent entries
            if source_log:
                target_log_path = target_path / "log.md"
                with open(target_log_path, 'a') as f:
                    f.write(f"\n## Copied from {source_branch}\n{source_log}\n")
        
        except Exception:
            pass  # Continue if copy fails
    
    def _create_git_commit(self, summary: str):
        """Create a git commit if git integration is enabled."""
        try:
            subprocess.run(["git", "add", str(self.gcc_path)], 
                         cwd=self.project_root, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"GCC: {summary}"], 
                         cwd=self.project_root, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Git not available or not a git repository - continue without git integration
            pass
    
    def _get_recent_commits(self, branch: str, count: int = 5) -> List[Dict[str, str]]:
        """Get recent commits for a branch."""
        commit_content = (self.gcc_path / "branches" / branch / "commit.md").read_text()
        
        commits = []
        lines = commit_content.split('\n')
        
        current_commit = {}
        for line in lines:
            if line.startswith('## Commit:'):
                if current_commit:
                    commits.append(current_commit)
                timestamp = line.replace('## Commit:', '').strip()
                current_commit = {'timestamp': timestamp, 'summary': ''}
            elif line.startswith('**Summary:**') and current_commit:
                current_commit['summary'] = line.replace('**Summary:**', '').strip()
        
        if current_commit:
            commits.append(current_commit)
        
        return commits[-count:] if len(commits) > count else commits
    
    def _compress_ota_log(self):
        """Compress current OTA log to prevent memory bloat."""
        if len(self.current_ota_log) <= 10:
            return
        
        # Keep first 5 and last 5, compress middle
        compressed_summary = f"... {len(self.current_ota_log) - 10} OTA steps compressed for space efficiency ..."
        compressed_ota = OTAStep(
            timestamp=datetime.now().isoformat(),
            observation="Compressed OTA log",
            thought="Log compressed to prevent memory bloat",
            action=compressed_summary,
            metadata={"compressed_count": len(self.current_ota_log) - 10}
        )
        
        self.current_ota_log = (
            self.current_ota_log[:5] + 
            [compressed_ota] + 
            self.current_ota_log[-5:]
        )