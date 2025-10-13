"""
Agent adapters for integrating GCC with different AI agent frameworks.

This module provides base classes and specific adapters for integrating
Git-Controlled Context with various agent systems like OpenAI, Anthropic,
LangChain, AutoGen, etc.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

from .core import GitContextController
from .commands import GCCCommandInterface


class BaseAgentAdapter(ABC):
    """
    Base adapter class for integrating GCC with AI agent frameworks.
    
    This abstract class defines the interface that specific agent adapters
    should implement to integrate with their respective frameworks.
    """
    
    def __init__(self, gcc_controller: GitContextController, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter.
        
        Args:
            gcc_controller: GitContextController instance
            config: Optional configuration dictionary
        """
        self.gcc_controller = gcc_controller
        self.gcc_commands = GCCCommandInterface(gcc_controller)
        self.config = config or {}
        
        # Adapter state
        self.current_ota = {"observation": "", "thought": "", "action": ""}
        self.auto_commit_threshold = self.config.get("auto_commit_threshold", 5)
        self.action_count = 0
        
    @abstractmethod
    def wrap_agent(self, agent) -> Any:
        """
        Wrap an existing agent with GCC capabilities.
        
        Args:
            agent: The agent instance to wrap
            
        Returns:
            Wrapped agent with GCC integration
        """
        pass
    
    @abstractmethod
    def extract_ota_from_response(self, response: str) -> Dict[str, str]:
        """
        Extract OTA (Observation-Thought-Action) components from agent response.
        
        Args:
            response: Agent's response text
            
        Returns:
            Dictionary with observation, thought, and action components
        """
        pass
    
    @abstractmethod
    def inject_gcc_instructions(self, prompt: str) -> str:
        """
        Inject GCC instructions into agent prompt.
        
        Args:
            prompt: Original agent prompt
            
        Returns:
            Enhanced prompt with GCC instructions
        """
        pass
    
    def extract_gcc_command(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract GCC command from agent response.
        
        Args:
            response: Agent's response text
            
        Returns:
            GCC command dictionary if found, None otherwise
        """
        # Look for JSON-formatted GCC commands
        json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        
        for json_block in json_blocks:
            try:
                command = json.loads(json_block)
                if self._is_gcc_command(command):
                    return command
            except json.JSONDecodeError:
                continue
        
        # Also check for direct JSON objects
        json_patterns = re.findall(r'\{[^{}]*"action"[^{}]*\}', response)
        for pattern in json_patterns:
            try:
                command = json.loads(pattern)
                if self._is_gcc_command(command):
                    return command
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _is_gcc_command(self, command: Dict[str, Any]) -> bool:
        """Check if a command is a valid GCC command."""
        gcc_actions = [
            "commit", "create_branch", "merge", "show_context",
            "switch_branch", "add_ota", "update_metadata"
        ]
        return command.get("action") in gcc_actions
    
    def execute_gcc_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GCC command and return results."""
        return self.gcc_commands.execute_command(command)
    
    def process_agent_response(self, response: str) -> Dict[str, Any]:
        """
        Process agent response for GCC integration.
        
        Args:
            response: Agent's response text
            
        Returns:
            Processing results including any GCC command results
        """
        result = {
            "original_response": response,
            "ota_extracted": False,
            "gcc_command_executed": False,
            "gcc_result": None,
            "processed_response": response
        }
        
        # Extract OTA components
        ota = self.extract_ota_from_response(response)
        if any(ota.values()):
            self.current_ota.update(ota)
            result["ota_extracted"] = True
            result["ota_components"] = ota
        
        # Check for GCC commands
        gcc_command = self.extract_gcc_command(response)
        if gcc_command:
            gcc_result = self.execute_gcc_command(gcc_command)
            result["gcc_command_executed"] = True
            result["gcc_result"] = gcc_result
            
            # Update processed response to include GCC result
            result["processed_response"] = f"{response}\n\nGCC Command Result:\n{json.dumps(gcc_result, indent=2)}"
        
        # Auto-commit logic
        self.action_count += 1
        if self.action_count >= self.auto_commit_threshold:
            self._auto_commit()
            result["auto_commit_triggered"] = True
            self.action_count = 0
        
        return result
    
    def _auto_commit(self):
        """Automatically commit progress."""
        if self.current_ota.get("action"):
            summary = f"Auto-commit: {self.current_ota['action'][:50]}..."
        else:
            summary = f"Auto-commit after {self.auto_commit_threshold} actions"
        
        # Add current OTA if available
        if any(self.current_ota.values()):
            self.gcc_controller.add_ota_step(
                self.current_ota.get("observation", ""),
                self.current_ota.get("thought", ""),
                self.current_ota.get("action", "")
            )
        
        self.gcc_controller.commit(summary)
        
        # Reset OTA
        self.current_ota = {"observation": "", "thought": "", "action": ""}
    
    def get_gcc_context_summary(self) -> str:
        """Get a summary of current GCC context for the agent."""
        context = self.gcc_controller.show_context()
        
        summary = f"""
GCC Context Summary:
- Current Branch: {context.get('current_branch', 'unknown')}
- Available Branches: {', '.join(context.get('available_branches', []))}
- Recent Activity: Available in commit history and execution logs
- Memory Structure: Organized in .GCC directory with structured metadata
"""
        return summary.strip()
    
    def get_gcc_instructions(self) -> str:
        """Get standard GCC instructions for agent prompts."""
        return """
## Git-Controlled Context (GCC) Integration

You have access to a structured memory system that works like Git for managing your reasoning:

### Available GCC Commands (use JSON format):

1. **Commit Progress**: Create checkpoint when reaching milestone
```json
{"action": "commit", "args": {"summary": "Implemented feature X"}}
```

2. **Create Branch**: Explore alternative approach
```json
{"action": "create_branch", "args": {"name": "alternative_approach", "goal": "Test different solution"}}
```

3. **Show Context**: Review project status and history
```json
{"action": "show_context", "args": {}}
```

4. **Switch Branch**: Change to different reasoning path
```json
{"action": "switch_branch", "args": {"branch": "branch_name"}}
```

### Structured Reasoning (OTA Format):
Use this pattern for clear reasoning:
- **OBSERVATION**: What you see/learn
- **THOUGHT**: What you infer/decide  
- **ACTION**: What you do next

### Strategy:
- Commit when you complete meaningful progress
- Branch when exploring alternatives
- Use context to recall previous work
- Structure your reasoning with OTA format
"""


class OpenAIAgentAdapter(BaseAgentAdapter):
    """Adapter for OpenAI-based agents."""
    
    def wrap_agent(self, agent) -> Any:
        """Wrap OpenAI agent with GCC capabilities."""
        
        # Store original methods
        original_chat = agent.chat if hasattr(agent, 'chat') else None
        original_complete = agent.complete if hasattr(agent, 'complete') else None
        
        def gcc_enhanced_chat(*args, **kwargs):
            # Inject GCC instructions into messages
            if args and isinstance(args[0], list):
                messages = args[0]
                if messages and messages[0].get('role') == 'system':
                    messages[0]['content'] = self.inject_gcc_instructions(messages[0]['content'])
            
            # Call original method
            response = original_chat(*args, **kwargs)
            
            # Process response for GCC
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                self.process_agent_response(content)
            
            return response
        
        def gcc_enhanced_complete(*args, **kwargs):
            # Inject GCC instructions into prompt
            if args:
                prompt = args[0]
                enhanced_prompt = self.inject_gcc_instructions(prompt)
                args = (enhanced_prompt,) + args[1:]
            
            # Call original method
            response = original_complete(*args, **kwargs)
            
            # Process response for GCC
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].text
                self.process_agent_response(content)
            
            return response
        
        # Replace methods
        if original_chat:
            agent.chat = gcc_enhanced_chat
        if original_complete:
            agent.complete = gcc_enhanced_complete
        
        return agent
    
    def extract_ota_from_response(self, response: str) -> Dict[str, str]:
        """Extract OTA components from OpenAI response."""
        ota = {"observation": "", "thought": "", "action": ""}
        
        # Look for explicit OTA structure
        obs_match = re.search(r"(?:OBSERVATION|Observation):\s*(.+?)(?=(?:THOUGHT|Thought|ACTION|Action):|$)", response, re.DOTALL | re.IGNORECASE)
        thought_match = re.search(r"(?:THOUGHT|Thought):\s*(.+?)(?=(?:ACTION|Action|OBSERVATION|Observation):|$)", response, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"(?:ACTION|Action):\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
        
        if obs_match:
            ota["observation"] = obs_match.group(1).strip()
        if thought_match:
            ota["thought"] = thought_match.group(1).strip()
        if action_match:
            ota["action"] = action_match.group(1).strip()
        
        return ota
    
    def inject_gcc_instructions(self, prompt: str) -> str:
        """Inject GCC instructions into OpenAI prompt."""
        gcc_instructions = self.get_gcc_instructions()
        return f"{prompt}\n\n{gcc_instructions}"


class AnthropicAgentAdapter(BaseAgentAdapter):
    """Adapter for Anthropic Claude-based agents."""
    
    def wrap_agent(self, agent) -> Any:
        """Wrap Anthropic agent with GCC capabilities."""
        
        original_complete = getattr(agent, 'complete', None) or getattr(agent, 'messages', None)
        
        def gcc_enhanced_complete(*args, **kwargs):
            # Inject GCC instructions
            if 'prompt' in kwargs:
                kwargs['prompt'] = self.inject_gcc_instructions(kwargs['prompt'])
            elif args:
                prompt = args[0]
                enhanced_prompt = self.inject_gcc_instructions(prompt)
                args = (enhanced_prompt,) + args[1:]
            
            # Call original method
            response = original_complete(*args, **kwargs)
            
            # Process response for GCC
            if hasattr(response, 'completion'):
                content = response.completion
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            self.process_agent_response(content)
            
            return response
        
        if original_complete:
            if hasattr(agent, 'complete'):
                agent.complete = gcc_enhanced_complete
            elif hasattr(agent, 'messages'):
                agent.messages = gcc_enhanced_complete
        
        return agent
    
    def extract_ota_from_response(self, response: str) -> Dict[str, str]:
        """Extract OTA components from Anthropic response."""
        return self._standard_ota_extraction(response)
    
    def inject_gcc_instructions(self, prompt: str) -> str:
        """Inject GCC instructions into Anthropic prompt."""
        gcc_instructions = self.get_gcc_instructions()
        return f"{prompt}\n\nHuman: {gcc_instructions}\n\nAssistant: I understand. I'll use the GCC system for structured memory management and reasoning. Let me proceed with the task using OTA format and GCC commands as needed."
    
    def _standard_ota_extraction(self, response: str) -> Dict[str, str]:
        """Standard OTA extraction logic."""
        ota = {"observation": "", "thought": "", "action": ""}
        
        patterns = [
            (r"(?:OBSERVATION|Observation):\s*(.+?)(?=(?:THOUGHT|Thought|ACTION|Action):|$)", "observation"),
            (r"(?:THOUGHT|Thought):\s*(.+?)(?=(?:ACTION|Action|OBSERVATION|Observation):|$)", "thought"),
            (r"(?:ACTION|Action):\s*(.+?)$", "action")
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                ota[key] = match.group(1).strip()
        
        return ota


class LangChainAgentAdapter(BaseAgentAdapter):
    """Adapter for LangChain-based agents."""
    
    def wrap_agent(self, agent) -> Any:
        """Wrap LangChain agent with GCC capabilities."""
        
        original_run = getattr(agent, 'run', None)
        original_call = getattr(agent, '__call__', None)
        
        def gcc_enhanced_run(*args, **kwargs):
            # Inject GCC context
            if args:
                query = args[0]
                enhanced_query = f"{query}\n\n{self.get_gcc_context_summary()}"
                args = (enhanced_query,) + args[1:]
            
            response = original_run(*args, **kwargs)
            self.process_agent_response(str(response))
            return response
        
        def gcc_enhanced_call(*args, **kwargs):
            response = original_call(*args, **kwargs)
            self.process_agent_response(str(response))
            return response
        
        if original_run:
            agent.run = gcc_enhanced_run
        if original_call:
            agent.__call__ = gcc_enhanced_call
        
        return agent
    
    def extract_ota_from_response(self, response: str) -> Dict[str, str]:
        """Extract OTA components from LangChain response."""
        return self._standard_ota_extraction(response)
    
    def inject_gcc_instructions(self, prompt: str) -> str:
        """Inject GCC instructions into LangChain prompt."""
        return f"{prompt}\n\n{self.get_gcc_instructions()}"
    
    def _standard_ota_extraction(self, response: str) -> Dict[str, str]:
        """Standard OTA extraction logic."""
        ota = {"observation": "", "thought": "", "action": ""}
        
        # LangChain often has structured output, try to parse it
        if "Thought:" in response or "Action:" in response:
            thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
            action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|Observation:|$)", response, re.DOTALL)
            obs_match = re.search(r"Observation:\s*(.+?)(?=Thought:|$)", response, re.DOTALL)
            
            if thought_match:
                ota["thought"] = thought_match.group(1).strip()
            if action_match:
                ota["action"] = action_match.group(1).strip()
            if obs_match:
                ota["observation"] = obs_match.group(1).strip()
        
        return ota


class GenericAgentAdapter(BaseAgentAdapter):
    """Generic adapter that can work with any agent framework."""
    
    def __init__(self, gcc_controller: GitContextController, config: Optional[Dict[str, Any]] = None):
        super().__init__(gcc_controller, config)
        
        # User-provided hooks
        self.response_processor: Optional[Callable[[str], Dict[str, str]]] = config.get("response_processor") if config else None
        self.prompt_enhancer: Optional[Callable[[str], str]] = config.get("prompt_enhancer") if config else None
    
    def wrap_agent(self, agent) -> Any:
        """Wrap any agent with GCC capabilities."""
        
        # Store reference to GCC adapter in agent
        agent._gcc_adapter = self
        
        # Add GCC methods to agent
        agent.gcc_commit = self.gcc_commands.commit
        agent.gcc_create_branch = self.gcc_commands.create_branch
        agent.gcc_show_context = self.gcc_commands.show_context
        agent.gcc_add_ota = self.gcc_commands.add_ota_step
        
        return agent
    
    def extract_ota_from_response(self, response: str) -> Dict[str, str]:
        """Extract OTA components from any agent response."""
        if self.response_processor:
            return self.response_processor(response)
        else:
            return self._standard_ota_extraction(response)
    
    def inject_gcc_instructions(self, prompt: str) -> str:
        """Inject GCC instructions into any prompt."""
        if self.prompt_enhancer:
            return self.prompt_enhancer(prompt)
        else:
            return f"{prompt}\n\n{self.get_gcc_instructions()}"
    
    def _standard_ota_extraction(self, response: str) -> Dict[str, str]:
        """Standard OTA extraction logic."""
        ota = {"observation": "", "thought": "", "action": ""}
        
        # Try multiple patterns
        patterns = [
            (r"(?:OBSERVATION|Observation|I observe):\s*(.+?)(?=(?:THOUGHT|Thought|I think|ACTION|Action|I will):|$)", "observation"),
            (r"(?:THOUGHT|Thought|I think):\s*(.+?)(?=(?:ACTION|Action|I will|OBSERVATION|Observation|I observe):|$)", "thought"),
            (r"(?:ACTION|Action|I will):\s*(.+?)$", "action")
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                ota[key] = match.group(1).strip()
        
        return ota


# Factory function for creating adapters
def create_adapter(framework: str, gcc_controller: GitContextController, config: Optional[Dict[str, Any]] = None) -> BaseAgentAdapter:
    """
    Factory function to create appropriate adapter for different frameworks.
    
    Args:
        framework: Framework name ('openai', 'anthropic', 'langchain', 'generic')
        gcc_controller: GitContextController instance
        config: Optional configuration
        
    Returns:
        Appropriate adapter instance
    """
    framework = framework.lower()
    
    if framework == 'openai':
        return OpenAIAgentAdapter(gcc_controller, config)
    elif framework in ['anthropic', 'claude']:
        return AnthropicAgentAdapter(gcc_controller, config)
    elif framework == 'langchain':
        return LangChainAgentAdapter(gcc_controller, config)
    elif framework == 'generic':
        return GenericAgentAdapter(gcc_controller, config)
    else:
        raise ValueError(f"Unsupported framework: {framework}. Use 'openai', 'anthropic', 'langchain', or 'generic'.")