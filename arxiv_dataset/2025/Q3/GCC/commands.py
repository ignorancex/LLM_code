"""
Command interface for Git-Context-Controller.

This module provides a high-level, framework-agnostic interface for AI agents
to interact with the GCC system using structured commands.
"""

import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

from core import GitContextController, GCCConfig


class GCCCommandInterface:
    """
    High-level command interface for Git-Context-Controller.
    
    This class provides a framework-agnostic way for AI agents to interact
    with the GCC system using structured JSON commands or method calls.
    """
    
    def __init__(self, controller: GitContextController):
        """
        Initialize the command interface.
        
        Args:
            controller: GitContextController instance
        """
        self.controller = controller
        self._command_handlers = {
            "commit": self._handle_commit,
            "create_branch": self._handle_create_branch,
            "merge": self._handle_merge, 
            "show_context": self._handle_show_context,
            "switch_branch": self._handle_switch_branch,
            "list_branches": self._handle_list_branches,
            "add_ota": self._handle_add_ota,
            "update_metadata": self._handle_update_metadata,
            "get_branch_context": self._handle_get_branch_context,
            "get_log": self._handle_get_log,
            "delete_branch": self._handle_delete_branch
        }
    
    def execute_command(self, command: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a GCC command from JSON string or dictionary.
        
        Args:
            command: JSON string or dictionary with action and args
            
        Returns:
            Dictionary with command result
        """
        try:
            if isinstance(command, str):
                command_dict = json.loads(command)
            else:
                command_dict = command
            
            action = command_dict.get("action")
            args = command_dict.get("args", {})
            
            if action not in self._command_handlers:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": list(self._command_handlers.keys())
                }
            
            result = self._command_handlers[action](**args)
            return {
                "success": True,
                "action": action,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {
                "success": False,
                "error": f"Invalid command format: {e}",
                "expected_format": {
                    "action": "command_name",
                    "args": {"arg1": "value1", "arg2": "value2"}
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command execution failed: {e}"
            }
    
    # High-level method interface (alternative to JSON commands)
    
    def commit(self, summary: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a commit checkpoint."""
        result = self.controller.commit(summary, metadata)
        return {
            "action": "commit",
            "commit_info": {
                "timestamp": result.timestamp,
                "summary": result.summary,
                "branch": result.branch,
                "ota_count": result.ota_count
            }
        }
    
    def create_branch(self, name: str, goal: str, copy_from: Optional[str] = None) -> Dict[str, Any]:
        """Create a new branch."""
        return self.controller.create_branch(name, goal, copy_from)
    
    def merge(self, branches: List[str], summary: str, target_branch: Optional[str] = None) -> Dict[str, Any]:
        """Merge multiple branches."""
        return self.controller.merge(branches, summary, target_branch)
    
    def show_context(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Show project context."""
        return self.controller.show_context(options)
    
    def add_ota_step(self, observation: str, thought: str, action: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an OTA step to current reasoning cycle."""
        self.controller.add_ota_step(observation, thought, action, metadata)
    
    def switch_branch(self, branch: str) -> Dict[str, Any]:
        """Switch to a different branch."""
        success = self.controller.switch_branch(branch)
        return {
            "action": "switch_branch",
            "success": success,
            "current_branch": self.controller.current_branch,
            "message": f"Switched to {branch}" if success else f"Branch {branch} does not exist"
        }
    
    def list_branches(self) -> Dict[str, Any]:
        """List all available branches."""
        branches = self.controller.list_branches()
        return {
            "action": "list_branches",
            "branches": branches,
            "current_branch": self.controller.current_branch,
            "count": len(branches)
        }
    
    def get_branch_context(self, branch: str, include_log: bool = False, log_lines: int = 20) -> Dict[str, Any]:
        """Get detailed context for a specific branch."""
        context = self.controller.get_branch_context(branch, include_log, log_lines)
        return {
            "action": "get_branch_context",
            "branch": branch,
            "context": context
        }
    
    def update_metadata(self, metadata: Dict[str, Any], branch: Optional[str] = None) -> Dict[str, Any]:
        """Update branch metadata."""
        self.controller.update_branch_metadata(metadata, branch)
        return {
            "action": "update_metadata",
            "branch": branch or self.controller.current_branch,
            "message": "Metadata updated successfully"
        }
    
    def delete_branch(self, branch: str, force: bool = False) -> Dict[str, Any]:
        """Delete a branch."""
        success = self.controller.delete_branch(branch, force)
        return {
            "action": "delete_branch",
            "success": success,
            "branch": branch,
            "message": f"Branch {branch} deleted" if success else f"Failed to delete branch {branch}"
        }
    
    # Command handlers for JSON interface
    
    def _handle_commit(self, summary: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle commit command."""
        return self.commit(summary, metadata)
    
    def _handle_create_branch(self, name: str, goal: str, copy_from: Optional[str] = None) -> Dict[str, Any]:
        """Handle create_branch command."""
        return self.create_branch(name, goal, copy_from)
    
    def _handle_merge(self, branches: List[str], summary: str, target_branch: Optional[str] = None) -> Dict[str, Any]:
        """Handle merge command."""
        return self.merge(branches, summary, target_branch)
    
    def _handle_show_context(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle show_context command."""
        return self.show_context(options)
    
    def _handle_switch_branch(self, branch: str) -> Dict[str, Any]:
        """Handle switch_branch command."""
        return self.switch_branch(branch)
    
    def _handle_list_branches(self) -> Dict[str, Any]:
        """Handle list_branches command."""
        return self.list_branches()
    
    def _handle_add_ota(self, observation: str, thought: str, action: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle add_ota command."""
        self.add_ota_step(observation, thought, action, metadata)
        return {
            "action": "add_ota",
            "message": "OTA step added successfully",
            "current_ota_count": len(self.controller.current_ota_log)
        }
    
    def _handle_update_metadata(self, metadata: Dict[str, Any], branch: Optional[str] = None) -> Dict[str, Any]:
        """Handle update_metadata command."""
        return self.update_metadata(metadata, branch)
    
    def _handle_get_branch_context(self, branch: str, include_log: bool = False, log_lines: int = 20) -> Dict[str, Any]:
        """Handle get_branch_context command."""
        return self.get_branch_context(branch, include_log, log_lines)
    
    def _handle_get_log(self, branch: Optional[str] = None, lines: Optional[int] = None) -> Dict[str, Any]:
        """Handle get_log command."""
        branch = branch or self.controller.current_branch
        log_content = self.controller.get_branch_log(branch, lines)
        return {
            "action": "get_log",
            "branch": branch,
            "log_content": log_content,
            "lines_requested": lines
        }
    
    def _handle_delete_branch(self, branch: str, force: bool = False) -> Dict[str, Any]:
        """Handle delete_branch command."""
        return self.delete_branch(branch, force)
    
    # Utility methods
    
    def get_command_help(self, command: Optional[str] = None) -> Dict[str, Any]:
        """Get help for available commands."""
        command_docs = {
            "commit": {
                "description": "Create a checkpoint when reaching stable milestone",
                "args": {
                    "summary": "Short summary of what was accomplished",
                    "metadata": "Optional metadata dictionary"
                },
                "example": {
                    "action": "commit",
                    "args": {"summary": "Implemented feature X"}
                }
            },
            "create_branch": {
                "description": "Create a new branch for alternative exploration",
                "args": {
                    "name": "Branch name",
                    "goal": "Purpose/goal of the branch",
                    "copy_from": "Optional branch to copy from"
                },
                "example": {
                    "action": "create_branch",
                    "args": {"name": "feature_x", "goal": "Implement feature X"}
                }
            },
            "merge": {
                "description": "Consolidate progress from multiple branches",
                "args": {
                    "branches": "List of branch names to merge",
                    "summary": "Summary of merged conclusions",
                    "target_branch": "Optional target branch"
                },
                "example": {
                    "action": "merge",
                    "args": {"branches": ["branch1", "branch2"], "summary": "Unified approaches"}
                }
            },
            "show_context": {
                "description": "Review project history and branch status",
                "args": {
                    "options": "Optional context retrieval options"
                },
                "example": {
                    "action": "show_context",
                    "args": {"options": {"include_metadata": True}}
                }
            },
            "switch_branch": {
                "description": "Switch to a different branch",
                "args": {
                    "branch": "Branch name to switch to"
                },
                "example": {
                    "action": "switch_branch",
                    "args": {"branch": "feature_branch"}
                }
            },
            "add_ota": {
                "description": "Add OTA step to current reasoning cycle",
                "args": {
                    "observation": "What was observed",
                    "thought": "What was thought/inferred",
                    "action": "What action was taken",
                    "metadata": "Optional metadata"
                },
                "example": {
                    "action": "add_ota",
                    "args": {
                        "observation": "Code compiles successfully",
                        "thought": "Ready to test functionality",
                        "action": "Running unit tests"
                    }
                }
            }
        }
        
        if command:
            if command in command_docs:
                return {
                    "command": command,
                    "documentation": command_docs[command]
                }
            else:
                return {
                    "error": f"Unknown command: {command}",
                    "available_commands": list(command_docs.keys())
                }
        
        return {
            "available_commands": list(command_docs.keys()),
            "documentation": command_docs,
            "usage": "Use execute_command() with JSON format or call methods directly"
        }
    
    def export_history(self, format: str = "json") -> Union[Dict[str, Any], str]:
        """Export complete GCC history."""
        context = self.controller.show_context({
            "include_metadata": True,
            "branch_details": True
        })
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "gcc_version": "1.0.0",
            "project_root": str(self.controller.project_root),
            "current_branch": self.controller.current_branch,
            "context": context
        }
        
        if format == "json":
            return export_data
        elif format == "yaml":
            import yaml
            return yaml.dump(export_data, default_flow_style=False)
        else:
            return json.dumps(export_data, indent=2)
    
    def validate_structure(self) -> Dict[str, Any]:
        """Validate GCC structure integrity."""
        validation_results = {
            "gcc_path_exists": self.controller.gcc_path.exists(),
            "main_md_exists": (self.controller.gcc_path / "main.md").exists(),
            "branches_dir_exists": (self.controller.gcc_path / "branches").exists(),
            "branch_validations": {},
            "issues": []
        }
        
        if validation_results["branches_dir_exists"]:
            branches = self.controller.list_branches()
            for branch in branches:
                branch_path = self.controller.gcc_path / "branches" / branch
                branch_validation = {
                    "commit_md_exists": (branch_path / "commit.md").exists(),
                    "log_md_exists": (branch_path / "log.md").exists(),
                    "metadata_yaml_exists": (branch_path / "metadata.yaml").exists()
                }
                
                if not all(branch_validation.values()):
                    validation_results["issues"].append(f"Branch {branch} missing required files")
                
                validation_results["branch_validations"][branch] = branch_validation
        
        validation_results["overall_valid"] = (
            validation_results["gcc_path_exists"] and
            validation_results["main_md_exists"] and 
            validation_results["branches_dir_exists"] and
            len(validation_results["issues"]) == 0
        )
        
        return validation_results