"""
Memory management utilities for Git-Context-Controller.

This module provides utilities for managing structured memory files,
performing operations across different granularity levels, and optimizing
memory usage in the GCC system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import re


class MemoryManager:
    """
    Manages memory files and structured data for GCC system.
    
    This class provides utilities for reading, writing, searching, and optimizing
    memory files across the GCC directory structure.
    """
    
    def __init__(self, gcc_path: Path):
        """
        Initialize the memory manager.
        
        Args:
            gcc_path: Path to the .GCC directory
        """
        self.gcc_path = gcc_path
    
    # High-level memory operations
    
    def read_main_roadmap(self) -> str:
        """Read the main project roadmap."""
        main_md = self.gcc_path / "main.md"
        if main_md.exists():
            return main_md.read_text()
        return ""
    
    def update_main_roadmap(self, content: str):
        """Update the main project roadmap."""
        main_md = self.gcc_path / "main.md"
        main_md.write_text(content)
    
    def get_branch_commit_history(self, branch: str) -> str:
        """Get commit history for a specific branch."""
        commit_md = self.gcc_path / "branches" / branch / "commit.md"
        if commit_md.exists():
            return commit_md.read_text()
        return ""
    
    def get_branch_log(self, branch: str, lines: Optional[int] = None) -> str:
        """Get execution log for a specific branch."""
        log_md = self.gcc_path / "branches" / branch / "log.md"
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
        metadata_yaml = self.gcc_path / "branches" / branch / "metadata.yaml"
        if metadata_yaml.exists():
            try:
                with open(metadata_yaml) as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError:
                return {}
        return {}
    
    def update_branch_metadata(self, branch: str, metadata: Dict[str, Any]):
        """Update metadata for a specific branch."""
        metadata_yaml = self.gcc_path / "branches" / branch / "metadata.yaml"
        
        # Deep merge with existing metadata
        existing = self.get_branch_metadata(branch)
        merged = self._deep_merge_dict(existing, metadata)
        merged["last_updated"] = datetime.now().isoformat()
        
        with open(metadata_yaml, 'w') as f:
            yaml.dump(merged, f, default_flow_style=False)
    
    def list_branches(self) -> List[str]:
        """List all available branches."""
        branches_dir = self.gcc_path / "branches"
        if branches_dir.exists():
            return [d.name for d in branches_dir.iterdir() if d.is_dir()]
        return []
    
    # Structured data extraction
    
    def get_commit_summary(self, branch: str, n_commits: int = 10) -> List[Dict[str, str]]:
        """Extract last N commit summaries from branch."""
        commit_content = self.get_branch_commit_history(branch)
        
        commits = []
        lines = commit_content.split('\n')
        
        current_commit = {}
        for line in lines:
            if line.startswith('## Commit:'):
                if current_commit:
                    commits.append(current_commit)
                timestamp = line.replace('## Commit:', '').strip()
                current_commit = {'timestamp': timestamp, 'summary': '', 'content': ''}
            elif line.startswith('**Summary:**') and current_commit:
                current_commit['summary'] = line.replace('**Summary:**', '').strip()
            elif current_commit:
                current_commit['content'] += line + '\n'
        
        if current_commit:
            commits.append(current_commit)
        
        return commits[-n_commits:] if len(commits) > n_commits else commits
    
    def get_ota_steps(self, branch: str, n_steps: int = 20) -> List[Dict[str, str]]:
        """Extract recent OTA steps from branch log."""
        log_content = self.get_branch_log(branch)
        
        # Find OTA step patterns
        ota_pattern = r'### Step \d+ \(([^)]+)\)\s*\*\*OBSERVATION:\*\* ([^\n]+)\s*\*\*THOUGHT:\*\* ([^\n]+)\s*\*\*ACTION:\*\* ([^\n]+)'
        matches = re.findall(ota_pattern, log_content, re.MULTILINE | re.DOTALL)
        
        ota_steps = []
        for match in matches[-n_steps:]:
            ota_steps.append({
                'timestamp': match[0],
                'observation': match[1].strip(),
                'thought': match[2].strip(),
                'action': match[3].strip()
            })
        
        return ota_steps
    
    def extract_key_decisions(self, branch: str) -> List[Dict[str, str]]:
        """Extract key decisions from branch history."""
        commits = self.get_commit_summary(branch)
        ota_steps = self.get_ota_steps(branch, 50)
        
        decisions = []
        
        # Extract from commits
        for commit in commits:
            if any(keyword in commit['summary'].lower() for keyword in ['decide', 'choose', 'select', 'resolve']):
                decisions.append({
                    'type': 'commit',
                    'timestamp': commit['timestamp'],
                    'decision': commit['summary'],
                    'context': commit['content'][:200] + '...'
                })
        
        # Extract from OTA steps
        for ota in ota_steps:
            if any(keyword in ota['thought'].lower() for keyword in ['decide', 'choose', 'will', 'should']):
                decisions.append({
                    'type': 'ota',
                    'timestamp': ota['timestamp'],
                    'decision': ota['thought'],
                    'context': f"Obs: {ota['observation'][:100]}... Action: {ota['action'][:100]}..."
                })
        
        # Sort by timestamp
        decisions.sort(key=lambda x: x['timestamp'])
        
        return decisions
    
    # Search and analysis
    
    def search_across_branches(self, query: str, case_sensitive: bool = False) -> Dict[str, List[Dict[str, str]]]:
        """Search for query across all branch contents."""
        results = {}
        
        if not case_sensitive:
            query = query.lower()
        
        for branch in self.list_branches():
            branch_results = []
            
            # Search in commit history
            commit_content = self.get_branch_commit_history(branch)
            search_content = commit_content if case_sensitive else commit_content.lower()
            if query in search_content:
                # Find specific lines with matches
                lines = commit_content.split('\n')
                for i, line in enumerate(lines):
                    search_line = line if case_sensitive else line.lower()
                    if query in search_line:
                        branch_results.append({
                            'type': 'commit',
                            'line_number': i + 1,
                            'content': line.strip(),
                            'context': self._get_line_context(lines, i, 2)
                        })
            
            # Search in logs
            log_content = self.get_branch_log(branch)
            search_content = log_content if case_sensitive else log_content.lower()
            if query in search_content:
                lines = log_content.split('\n')
                for i, line in enumerate(lines):
                    search_line = line if case_sensitive else line.lower()
                    if query in search_line:
                        branch_results.append({
                            'type': 'log',
                            'line_number': i + 1,
                            'content': line.strip(),
                            'context': self._get_line_context(lines, i, 2)
                        })
            
            # Search in metadata
            metadata = self.get_branch_metadata(branch)
            metadata_str = yaml.dump(metadata)
            search_content = metadata_str if case_sensitive else metadata_str.lower()
            if query in search_content:
                branch_results.append({
                    'type': 'metadata',
                    'content': f"Found in metadata: {query}",
                    'context': json.dumps(metadata, indent=2)[:500] + '...'
                })
            
            if branch_results:
                results[branch] = branch_results
        
        return results
    
    def analyze_branch_progression(self, branch: str) -> Dict[str, Any]:
        """Analyze progression and patterns in a branch."""
        commits = self.get_commit_summary(branch)
        ota_steps = self.get_ota_steps(branch, 100)
        metadata = self.get_branch_metadata(branch)
        
        analysis = {
            'branch': branch,
            'total_commits': len(commits),
            'total_ota_steps': len(ota_steps),
            'time_span': {},
            'activity_patterns': {},
            'key_themes': [],
            'progress_indicators': []
        }
        
        # Time span analysis
        if commits:
            first_commit = commits[0]['timestamp']
            last_commit = commits[-1]['timestamp']
            analysis['time_span'] = {
                'first_activity': first_commit,
                'last_activity': last_commit,
                'duration_days': self._calculate_duration_days(first_commit, last_commit)
            }
        
        # Activity patterns
        commit_summaries = [c['summary'] for c in commits]
        ota_actions = [ota['action'] for ota in ota_steps]
        
        analysis['activity_patterns'] = {
            'most_common_commit_words': self._extract_common_words(commit_summaries),
            'most_common_action_words': self._extract_common_words(ota_actions),
            'avg_ota_per_commit': len(ota_steps) / len(commits) if commits else 0
        }
        
        # Key themes (simple keyword analysis)
        all_text = ' '.join(commit_summaries + ota_actions)
        analysis['key_themes'] = self._extract_key_themes(all_text)
        
        # Progress indicators
        progress_keywords = ['complete', 'finish', 'done', 'implement', 'fix', 'resolve', 'add', 'create']
        progress_count = sum(1 for summary in commit_summaries 
                           if any(keyword in summary.lower() for keyword in progress_keywords))
        
        analysis['progress_indicators'] = {
            'completion_commits': progress_count,
            'completion_ratio': progress_count / len(commits) if commits else 0,
            'recent_activity': len([c for c in commits if self._is_recent(c['timestamp'], days=7)])
        }
        
        return analysis
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        stats = {
            'total_branches': len(self.list_branches()),
            'branch_stats': {},
            'global_stats': {
                'total_commits': 0,
                'total_ota_steps': 0,
                'total_file_size_bytes': 0
            }
        }
        
        main_md = self.gcc_path / "main.md"
        if main_md.exists():
            stats['global_stats']['main_roadmap_size'] = main_md.stat().st_size
        
        for branch in self.list_branches():
            branch_path = self.gcc_path / "branches" / branch
            
            branch_stat = {
                'commit_count': len(self.get_commit_summary(branch)),
                'ota_step_count': len(self.get_ota_steps(branch, 1000)),  # Get all
                'file_sizes': {}
            }
            
            for filename in ['commit.md', 'log.md', 'metadata.yaml']:
                file_path = branch_path / filename
                if file_path.exists():
                    size = file_path.stat().st_size
                    branch_stat['file_sizes'][filename] = size
                    stats['global_stats']['total_file_size_bytes'] += size
            
            branch_stat['total_size_bytes'] = sum(branch_stat['file_sizes'].values())
            
            stats['branch_stats'][branch] = branch_stat
            stats['global_stats']['total_commits'] += branch_stat['commit_count']
            stats['global_stats']['total_ota_steps'] += branch_stat['ota_step_count']
        
        return stats
    
    # Memory optimization
    
    def compress_branch_history(self, branch: str, keep_recent: int = 10) -> Dict[str, Any]:
        """Compress older commit history to save space."""
        commits = self.get_commit_summary(branch, 1000)  # Get all commits
        
        if len(commits) <= keep_recent:
            return {'compressed': False, 'reason': 'Not enough commits to compress'}
        
        # Keep recent commits, compress older ones
        recent_commits = commits[-keep_recent:]
        older_commits = commits[:-keep_recent]
        
        # Create compressed summary
        compressed_info = {
            'compressed_commits': len(older_commits),
            'time_period': f"{older_commits[0]['timestamp']} to {older_commits[-1]['timestamp']}",
            'key_summaries': [c['summary'] for c in older_commits[::max(1, len(older_commits)//5)]]  # Sample
        }
        
        compressed_summary = f"""## Compressed History ({len(older_commits)} commits)
**Period:** {compressed_info['time_period']}
**Compressed commits:** {len(older_commits)}
**Key activities:** {', '.join(compressed_info['key_summaries'][:5])}
**Note:** History compressed for space efficiency on {datetime.now().isoformat()}

"""
        
        # Rebuild commit.md
        branch_path = self.gcc_path / "branches" / branch
        commit_md = branch_path / "commit.md"
        
        new_content = f"# Branch: {branch}\n\n{compressed_summary}"
        
        for commit in recent_commits:
            new_content += f"## Commit: {commit['timestamp']}\n"
            new_content += f"**Summary:** {commit['summary']}\n"
            new_content += commit['content']
        
        commit_md.write_text(new_content)
        
        return {
            'compressed': True,
            'compressed_commits': len(older_commits),
            'kept_recent': len(recent_commits),
            'compression_info': compressed_info
        }
    
    def cleanup_branch_logs(self, branch: str, max_ota_steps: int = 500) -> Dict[str, Any]:
        """Clean up branch logs by removing oldest OTA steps."""
        log_content = self.get_branch_log(branch)
        
        # Count current OTA steps
        ota_count = log_content.count('### Step')
        
        if ota_count <= max_ota_steps:
            return {'cleaned': False, 'reason': 'Log size within limits'}
        
        # Keep only recent OTA cycles
        cycles = log_content.split('## OTA Cycle:')
        if len(cycles) <= 1:
            return {'cleaned': False, 'reason': 'No OTA cycles found'}
        
        # Keep header and recent cycles
        header = cycles[0]
        recent_cycles = cycles[-10:]  # Keep last 10 cycles
        
        # Add compression note
        compression_note = f"""
## Log Compression Notice
Previous log entries were compressed on {datetime.now().isoformat()}.
Removed {len(cycles) - 10} older OTA cycles to optimize memory usage.
Total removed steps: approximately {ota_count - max_ota_steps}

"""
        
        new_content = header + compression_note + '## OTA Cycle:'.join(recent_cycles)
        
        # Write back to file
        log_md = self.gcc_path / "branches" / branch / "log.md"
        log_md.write_text(new_content)
        
        return {
            'cleaned': True,
            'removed_cycles': len(cycles) - 10,
            'remaining_cycles': 10,
            'estimated_steps_removed': ota_count - max_ota_steps
        }
    
    # Private utility methods
    
    def _deep_merge_dict(self, target: Dict, source: Dict) -> Dict:
        """Deep merge source dict into target dict."""
        result = target.copy()
        for key, value in source.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        return result
    
    def _get_line_context(self, lines: List[str], line_index: int, context_size: int) -> str:
        """Get context around a specific line."""
        start = max(0, line_index - context_size)
        end = min(len(lines), line_index + context_size + 1)
        context_lines = lines[start:end]
        
        # Mark the target line
        if context_size < len(context_lines):
            target_index = line_index - start
            if 0 <= target_index < len(context_lines):
                context_lines[target_index] = f">>> {context_lines[target_index]}"
        
        return '\n'.join(context_lines)
    
    def _extract_common_words(self, texts: List[str], top_n: int = 5) -> List[Tuple[str, int]]:
        """Extract most common words from texts."""
        word_counts = {}
        
        for text in texts:
            # Simple word extraction (could use more sophisticated NLP)
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top N words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
    
    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from text using simple keyword analysis."""
        # Common development themes
        theme_keywords = {
            'implementation': ['implement', 'code', 'develop', 'build', 'create'],
            'testing': ['test', 'verify', 'check', 'validate'],
            'debugging': ['debug', 'fix', 'error', 'issue', 'problem'],
            'optimization': ['optimize', 'improve', 'performance', 'efficient'],
            'refactoring': ['refactor', 'restructure', 'reorganize', 'clean'],
            'documentation': ['document', 'comment', 'explain', 'describe']
        }
        
        text_lower = text.lower()
        themes = []
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _calculate_duration_days(self, start_time: str, end_time: str) -> int:
        """Calculate duration in days between two timestamps."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).days
        except:
            return 0
    
    def _is_recent(self, timestamp: str, days: int = 7) -> bool:
        """Check if timestamp is within recent days."""
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
            return (now - ts).days <= days
        except:
            return False