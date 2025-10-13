"""
Git-Controlled Context (GCC) - A standalone context management system for AI agents.

This package provides version-controlled memory management for long-horizon reasoning
and development workflows, inspired by Git's approach to managing code history.

Key Features:
- Multi-level context retrieval (high-level plans to fine-grained execution traces)
- Isolated exploration via branching for parallel reasoning paths
- Structured reflection without re-parsing through clean memory organization
- Cross-agent flexibility for seamless handoffs and collaboration

The system is framework-agnostic and can be integrated with any AI agent system.
"""

__version__ = "1.0.0"
__author__ = "GCC Development Team"
__license__ = "MIT"

from .core import GitContextController
from .commands import GCCCommandInterface
from .memory import MemoryManager
from .adapters import BaseAgentAdapter
from .utils import GCCUtils

__all__ = [
    "GitContextController",
    "GCCCommandInterface", 
    "MemoryManager",
    "BaseAgentAdapter",
    "GCCUtils",
    "__version__"
]