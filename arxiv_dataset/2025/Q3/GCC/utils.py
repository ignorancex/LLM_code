"""
Utility functions for Git-Context-Controller operations.

This module provides various utility functions for GCC operations,
including validation, formatting, analysis, and system maintenance.
"""

import json
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import subprocess


class GCCUtils:
    """Utility functions for GCC operations."""
    
    @staticmethod
    def validate_gcc_structure(gcc_path: Path) -> Dict[str, Any]:
        """
        Validate GCC directory structure integrity.
        
        Args:
            gcc_path: Path to .GCC directory
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "gcc_dir_exists": gcc_path.exists(),
            "main_md_exists": (gcc_path / "main.md").exists(),
            "branches_dir_exists": (gcc_path / "branches").exists(),
            "branches": {},
            "issues": [],
            "warnings": []
        }
        
        if not validation["gcc_dir_exists"]:
            validation["issues"].append("GCC directory does not exist")
            return validation
        
        if not validation["main_md_exists"]:
            validation["issues"].append("main.md file missing")
        
        if not validation["branches_dir_exists"]:
            validation["issues"].append("branches directory missing")
            return validation
        
        # Validate each branch
        branches_dir = gcc_path / "branches"
        for branch_dir in branches_dir.iterdir():
            if not branch_dir.is_dir():
                continue
            
            branch_name = branch_dir.name
            branch_validation = {
                "commit_md_exists": (branch_dir / "commit.md").exists(),
                "log_md_exists": (branch_dir / "log.md").exists(),
                "metadata_yaml_exists": (branch_dir / "metadata.yaml").exists(),
                "metadata_valid": False
            }
            
            # Validate metadata YAML
            metadata_file = branch_dir / "metadata.yaml"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        yaml.safe_load(f)
                    branch_validation["metadata_valid"] = True
                except yaml.YAMLError as e:
                    validation["issues"].append(f"Branch {branch_name}: Invalid metadata YAML - {e}")
            
            # Check for missing files
            required_files = ["commit.md", "log.md", "metadata.yaml"]
            missing_files = [f for f in required_files if not (branch_dir / f).exists()]
            if missing_files:
                validation["issues"].append(f"Branch {branch_name}: Missing files - {missing_files}")
            
            validation["branches"][branch_name] = branch_validation
        
        # Overall validation
        validation["overall_valid"] = (
            validation["gcc_dir_exists"] and
            validation["main_md_exists"] and 
            validation["branches_dir_exists"] and
            len(validation["issues"]) == 0
        )
        
        return validation
    
    @staticmethod
    def repair_gcc_structure(gcc_path: Path) -> Dict[str, Any]:
        """
        Repair corrupted GCC structure.
        
        Args:
            gcc_path: Path to .GCC directory
            
        Returns:
            Dictionary with repair results
        """
        repair_log = {
            "repairs_made": [],
            "errors": [],
            "success": True
        }
        
        try:
            # Ensure main directories exist
            gcc_path.mkdir(exist_ok=True)
            repair_log["repairs_made"].append("Created GCC directory")
            
            (gcc_path / "branches").mkdir(exist_ok=True)
            repair_log["repairs_made"].append("Created branches directory")
            
            # Ensure main.md exists
            main_md = gcc_path / "main.md"
            if not main_md.exists():
                default_content = GCCUtils._get_default_main_content()
                main_md.write_text(default_content)
                repair_log["repairs_made"].append("Created main.md")
            
            # Repair branches
            branches_dir = gcc_path / "branches"
            for branch_dir in branches_dir.iterdir():
                if not branch_dir.is_dir():
                    continue
                
                branch_name = branch_dir.name
                repairs = GCCUtils._repair_branch_structure(branch_dir, branch_name)
                repair_log["repairs_made"].extend(repairs)
            
        except Exception as e:
            repair_log["errors"].append(f"Repair failed: {e}")
            repair_log["success"] = False
        
        return repair_log
    
    @staticmethod
    def _repair_branch_structure(branch_path: Path, branch_name: str) -> List[str]:
        """Repair individual branch structure."""
        repairs = []
        
        # Ensure commit.md exists
        commit_md = branch_path / "commit.md"
        if not commit_md.exists():
            commit_md.write_text(f"# Branch: {branch_name}\n\n## Commit History\n\n")
            repairs.append(f"Created commit.md for branch {branch_name}")
        
        # Ensure log.md exists
        log_md = branch_path / "log.md"
        if not log_md.exists():
            log_md.write_text("# OTA Execution Log\n\n")
            repairs.append(f"Created log.md for branch {branch_name}")
        
        # Ensure metadata.yaml exists
        metadata_yaml = branch_path / "metadata.yaml"
        if not metadata_yaml.exists():
            default_metadata = {
                "branch_info": {
                    "name": branch_name,
                    "created": datetime.now().isoformat(),
                    "purpose": "Branch restored during repair"
                },
                "file_structure": {},
                "env_config": {},
                "dependencies": [],
                "custom_data": {}
            }
            with open(metadata_yaml, 'w') as f:
                yaml.dump(default_metadata, f, default_flow_style=False)
            repairs.append(f"Created metadata.yaml for branch {branch_name}")
        
        return repairs
    
    @staticmethod
    def _get_default_main_content() -> str:
        """Get default content for main.md."""
        return f"""# Project Roadmap

## High-level Intent
This project uses Git-Controlled Context (GCC) for structured memory management.

## Milestones
- [x] GCC system initialization
- [ ] Core functionality implementation
- [ ] Testing and validation
- [ ] Documentation completion

## Shared Planning State
GCC system repaired and ready for structured reasoning.

Last updated: {datetime.now().isoformat()}
"""
    
    @staticmethod
    def parse_context_options(options_str: str) -> Dict[str, Any]:
        """
        Parse context retrieval options string.
        
        Args:
            options_str: Options string (e.g., "--branch main --log 20")
            
        Returns:
            Dictionary with parsed options
        """
        parsed = {
            "branch": None,
            "commit": None,
            "log_lines": None,
            "metadata_segment": None,
            "include_metadata": False,
            "include_log": False
        }
        
        if not options_str:
            return parsed
        
        parts = options_str.split()
        i = 0
        while i < len(parts):
            if parts[i] == "--branch" and i + 1 < len(parts):
                parsed["branch"] = parts[i + 1]
                i += 2
            elif parts[i] == "--commit" and i + 1 < len(parts):
                parsed["commit"] = parts[i + 1]
                i += 2
            elif parts[i] == "--log" and i + 1 < len(parts):
                try:
                    parsed["log_lines"] = int(parts[i + 1])
                    parsed["include_log"] = True
                except ValueError:
                    parsed["log_lines"] = 20
                i += 2
            elif parts[i] == "--metadata" and i + 1 < len(parts):
                parsed["metadata_segment"] = parts[i + 1]
                parsed["include_metadata"] = True
                i += 2
            elif parts[i] == "--full":
                parsed["include_metadata"] = True
                parsed["include_log"] = True
                parsed["log_lines"] = 50
                i += 1
            else:
                i += 1
        
        return parsed
    
    @staticmethod
    def format_context_response(context_data: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        Format context response for agent consumption.
        
        Args:
            context_data: Raw context data
            options: Parsed context options
            
        Returns:
            Formatted context string
        """
        if options.get("branch"):
            # Branch-specific context
            branch = options["branch"]
            output = f"## Branch: {branch}\n\n"
            
            if "context" in context_data and isinstance(context_data["context"], dict):
                branch_context = context_data["context"]
                
                if "commit_history" in branch_context:
                    output += "### Commit History\n"
                    output += branch_context["commit_history"][:1000] + "...\n\n"
                
                if "execution_log" in branch_context:
                    output += "### Recent Execution Log\n"
                    output += branch_context["execution_log"][:1000] + "...\n\n"
                
                if "metadata" in branch_context:
                    output += "### Metadata\n"
                    output += yaml.dump(branch_context["metadata"], default_flow_style=False)
            
            return output
        
        # Default overview format
        output = "# GCC Context Overview\n\n"
        
        if "main_roadmap" in context_data:
            output += "## Project Roadmap\n"
            roadmap = context_data["main_roadmap"][:800]
            output += roadmap + ("..." if len(context_data["main_roadmap"]) > 800 else "") + "\n\n"
        
        if "available_branches" in context_data:
            output += f"## Branches ({len(context_data['available_branches'])})\n"
            output += "- " + "\n- ".join(context_data["available_branches"]) + "\n\n"
        
        if "current_branch" in context_data:
            output += f"**Current Branch:** {context_data['current_branch']}\n\n"
        
        if "current_commit_history" in context_data:
            output += "## Recent Activity\n"
            history = context_data["current_commit_history"]
            # Extract last few commits
            commits = GCCUtils._extract_recent_commits(history, 3)
            for commit in commits:
                output += f"- {commit['timestamp']}: {commit['summary']}\n"
        
        return output
    
    @staticmethod
    def _extract_recent_commits(commit_history: str, n: int = 3) -> List[Dict[str, str]]:
        """Extract recent commits from commit history."""
        commits = []
        lines = commit_history.split('\n')
        
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
        
        return commits[-n:] if len(commits) > n else commits
    
    @staticmethod
    def detect_branch_suggestion(response_content: str) -> Optional[Tuple[str, str]]:
        """
        Detect if response suggests creating a new branch.
        
        Args:
            response_content: Agent's response content
            
        Returns:
            Tuple of (branch_name, goal) if branch should be created, None otherwise
        """
        branch_indicators = [
            r"(?:let me|i'll|i will) try (?:a |an )?(?:different|alternative|new) (?:approach|strategy|method|solution)",
            r"(?:alternative|different) (?:approach|solution|strategy|way)",
            r"(?:explore|try|test) (?:another|different|alternative)",
            r"(?:parallel|concurrent) (?:approach|development|work)",
            r"(?:branch|fork) (?:to|for) (?:explore|try|test)"
        ]
        
        content_lower = response_content.lower()
        
        for pattern in branch_indicators:
            if re.search(pattern, content_lower):
                # Generate branch name and goal
                timestamp = datetime.now().strftime("%m%d_%H%M")
                branch_name = f"exploration_{timestamp}"
                
                # Try to extract the specific approach mentioned
                approach_match = re.search(r"(?:try|explore|test) (?:a |an )?(.+?)(?:\.|,|$)", content_lower)
                if approach_match:
                    goal = f"Explore: {approach_match.group(1).strip()}"
                else:
                    goal = "Explore alternative approach based on current reasoning"
                
                return branch_name, goal
        
        return None
    
    @staticmethod
    def detect_merge_opportunity(gcc_controller) -> Optional[List[str]]:
        """
        Detect if branches should be merged.
        
        Args:
            gcc_controller: GitContextController instance
            
        Returns:
            List of branch names that could be merged
        """
        branches = gcc_controller.list_branches()
        main_branch = gcc_controller.config.default_branch
        
        # Simple heuristic: suggest merge if there are feature branches
        feature_branches = [b for b in branches if b != main_branch and b.startswith('exploration_')]
        
        if len(feature_branches) >= 2:
            return feature_branches
        elif len(feature_branches) == 1:
            # Check if the feature branch has meaningful commits
            from .memory import MemoryManager
            memory = MemoryManager(gcc_controller.gcc_path)
            commits = memory.get_commit_summary(feature_branches[0])
            if len(commits) >= 2:  # Has made some progress
                return feature_branches
        
        return None
    
    @staticmethod
    def generate_commit_message(ota_steps: List[Dict[str, str]], context: str = "") -> str:
        """
        Generate commit message from OTA steps and context.
        
        Args:
            ota_steps: List of OTA step dictionaries
            context: Additional context
            
        Returns:
            Generated commit message
        """
        if not ota_steps:
            return "Progress checkpoint"
        
        # Extract key actions and thoughts
        actions = [step.get("action", "") for step in ota_steps if step.get("action")]
        thoughts = [step.get("thought", "") for step in ota_steps if step.get("thought")]
        
        # Identify main theme
        all_text = " ".join(actions + thoughts).lower()
        
        themes = {
            "implementation": ["implement", "create", "build", "develop", "code"],
            "testing": ["test", "verify", "check", "validate"],
            "debugging": ["debug", "fix", "resolve", "error", "issue"],
            "optimization": ["optimize", "improve", "enhance", "performance"],
            "analysis": ["analyze", "examine", "study", "investigate"],
            "planning": ["plan", "design", "structure", "organize"]
        }
        
        detected_theme = None
        for theme, keywords in themes.items():
            if any(keyword in all_text for keyword in keywords):
                detected_theme = theme
                break
        
        # Generate message based on theme and actions
        if detected_theme:
            if len(actions) == 1:
                return f"{detected_theme.title()}: {actions[0][:50]}"
            else:
                return f"{detected_theme.title()} phase with {len(ota_steps)} steps"
        else:
            if len(actions) == 1:
                return f"Progress: {actions[0][:60]}"
            else:
                return f"Multi-step progress: {len(ota_steps)} actions completed"
    
    @staticmethod
    def extract_file_structure(project_root: Path, max_depth: int = 3) -> Dict[str, Any]:
        """
        Extract current file structure for metadata.
        
        Args:
            project_root: Root directory path
            max_depth: Maximum directory depth to scan
            
        Returns:
            Dictionary representing file structure
        """
        def scan_directory(path: Path, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth >= max_depth:
                return {"truncated": True, "reason": "max_depth_reached"}
            
            structure = {}
            try:
                for item in path.iterdir():
                    # Skip hidden files and common ignored directories
                    if item.name.startswith('.') and item.name not in ['.GCC']:
                        continue
                    
                    if item.name in ['node_modules', '__pycache__', '.git', 'venv', 'env']:
                        continue
                    
                    if item.is_file():
                        structure[item.name] = {
                            "type": "file",
                            "size": item.stat().st_size,
                            "extension": item.suffix,
                            "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                        }
                    elif item.is_dir():
                        structure[item.name] = {
                            "type": "directory",
                            "contents": scan_directory(item, current_depth + 1)
                        }
            except PermissionError:
                structure["_permission_denied"] = True
            
            return structure
        
        return {
            "scanned_at": datetime.now().isoformat(),
            "root_path": str(project_root),
            "max_depth": max_depth,
            "structure": scan_directory(project_root)
        }
    
    @staticmethod
    def check_git_integration(project_root: Path) -> Dict[str, Any]:
        """
        Check if Git integration is available and working.
        
        Args:
            project_root: Project root directory
            
        Returns:
            Dictionary with Git integration status
        """
        status = {
            "git_available": False,
            "is_git_repo": False,
            "can_commit": False,
            "current_branch": None,
            "uncommitted_changes": False,
            "error": None
        }
        
        try:
            # Check if git is available
            subprocess.run(["git", "--version"], 
                         capture_output=True, check=True, cwd=project_root)
            status["git_available"] = True
            
            # Check if it's a git repository
            result = subprocess.run(["git", "rev-parse", "--git-dir"], 
                                  capture_output=True, check=True, cwd=project_root)
            status["is_git_repo"] = True
            
            # Get current branch
            result = subprocess.run(["git", "branch", "--show-current"], 
                                  capture_output=True, check=True, cwd=project_root, text=True)
            status["current_branch"] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  capture_output=True, check=True, cwd=project_root, text=True)
            status["uncommitted_changes"] = bool(result.stdout.strip())
            
            # Test if we can create commits (check for user config)
            subprocess.run(["git", "config", "user.name"], 
                         capture_output=True, check=True, cwd=project_root)
            subprocess.run(["git", "config", "user.email"], 
                         capture_output=True, check=True, cwd=project_root)
            status["can_commit"] = True
            
        except subprocess.CalledProcessError as e:
            status["error"] = f"Git command failed: {e}"
        except FileNotFoundError:
            status["error"] = "Git not found in PATH"
        except Exception as e:
            status["error"] = f"Unexpected error: {e}"
        
        return status
    
    @staticmethod
    def export_gcc_data(gcc_controller, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export complete GCC data for backup or transfer.
        
        Args:
            gcc_controller: GitContextController instance
            format: Export format ('json', 'yaml', or 'dict')
            
        Returns:
            Exported data in requested format
        """
        from .memory import MemoryManager
        
        memory = MemoryManager(gcc_controller.gcc_path)
        
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "gcc_version": "1.0.0",
                "project_root": str(gcc_controller.project_root),
                "current_branch": gcc_controller.current_branch
            },
            "main_roadmap": memory.read_main_roadmap(),
            "branches": {}
        }
        
        # Export each branch
        for branch in memory.list_branches():
            branch_data = {
                "commit_history": memory.get_branch_commit_history(branch),
                "execution_log": memory.get_branch_log(branch),
                "metadata": memory.get_branch_metadata(branch),
                "commits_summary": memory.get_commit_summary(branch, 100),
                "ota_steps": memory.get_ota_steps(branch, 200)
            }
            export_data["branches"][branch] = branch_data
        
        # Add statistics
        export_data["statistics"] = memory.get_memory_statistics()
        
        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        elif format == "yaml":
            return yaml.dump(export_data, default_flow_style=False, default=str)
        else:  # dict
            return export_data