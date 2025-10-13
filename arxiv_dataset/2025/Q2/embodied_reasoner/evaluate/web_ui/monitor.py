#!/usr/bin/env python3

import asyncio
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from fastapi import WebSocket


class AgentMonitor:    
    def __init__(self):
        self.current_task = None
        self.task_history = []
        self.interaction_log = deque(maxlen=20)  # Recent 20 interactions  
        self.active_connections: List[WebSocket] = []
        self.disambiguation_active = False
        self.disambiguation_data = None
        self.user_selection = None
        self.disambiguation_history = []  # Complete history of all disambiguations
        self.task_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'current_task_index': 0
        }
   
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send current state
        await self.broadcast_state_update()
   
    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
   
    async def broadcast_state_update(self):
        """Broadcast state update to all connected clients"""
        if not self.active_connections:
            return
           
        state_data = {
            'type': 'state_update',
            'current_task': self.current_task,
            'task_history': list(self.task_history),
            'interaction_log': list(self.interaction_log),
            'task_stats': self.task_stats,
            'disambiguation_active': self.disambiguation_active,
            'disambiguation_data': self.disambiguation_data,
            'disambiguation_history': list(self.disambiguation_history),
            'timestamp': datetime.now().isoformat()
        }
       
        message = json.dumps(state_data, ensure_ascii=False)
        disconnected = []
       
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
       
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
   
    def _schedule_broadcast(self):
        """Safely schedule broadcast updates - simplified version"""
        if not self.active_connections:
            return  # No connected clients, no need to broadcast
           
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self.broadcast_state_update())
            return
        except RuntimeError:
            # No event loop, use a simplified thread pool method
            import concurrent.futures
           
            # Use a thread pool executor to avoid creating too many threads
            if not hasattr(self, '_executor'):
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="dashboard-broadcast")
           
            def broadcast_sync():
                try:
                    # Create a temporary event loop
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(self.broadcast_state_update())
                    finally:
                        loop.close()
                except Exception as e:
                    print(f"Push failed: {e}")
           
            # Submit to thread pool
            self._executor.submit(broadcast_sync)
   
    def start_task(self, task_data: Dict):
        """Start a new task"""
        self.current_task = {
            'id': task_data.get('identity', 'unknown'),
            'name': task_data.get('taskquery', 'Unknown task'),
            'scene': task_data.get('scene', 'Unknown'),
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'step_count': 0,
            'max_steps': task_data.get('max_steps', 20)
        }
       
        self.task_stats['current_task_index'] += 1
       
        # Clear interaction log
        self.interaction_log.clear()
        self.disambiguation_active = False
        self.disambiguation_data = None
       
        # Schedule broadcast if event loop is running
        self._schedule_broadcast()
   
    def add_interaction(self, interaction_data: Dict):
        """Add interaction log entry"""
        interaction = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': interaction_data.get('type', 'action'),
            'action': interaction_data.get('action', ''),
            'content': interaction_data.get('content', ''),
            'image_path': interaction_data.get('image_path', ''),
            'step': interaction_data.get('step', 0)
        }
       
        self.interaction_log.append(interaction)
       
        # Update current task step count
        if self.current_task:
            self.current_task['step_count'] = interaction.get('step', 0)
       
        # Schedule broadcast if event loop is running
        self._schedule_broadcast()
   
    def complete_task(self, success: bool, result_data: Dict = None):
        """Complete the current task"""
        if not self.current_task:
            return
           
        self.current_task.update({
            'status': 'completed' if success else 'failed',
            'end_time': datetime.now().isoformat(),
            'success': success,
            'result': result_data or {}
        })
       
        # Add to history
        self.task_history.append(self.current_task.copy())
       
        # Update stats
        self.task_stats['total_tasks'] = len(self.task_history)
        if success:
            self.task_stats['completed_tasks'] += 1
        else:
            self.task_stats['failed_tasks'] += 1
           
        # Schedule broadcast if event loop is running
        self._schedule_broadcast()
       
        # Clear current task
        self.current_task = None
   
    def start_disambiguation(self, disambiguation_data: Dict):
        """Start multi-object disambiguation"""
        self.disambiguation_active = True
        self.disambiguation_data = disambiguation_data
        self.user_selection = None
        
        # Add enhanced data for better tracking
        disambiguation_data['start_time'] = datetime.now().isoformat()
        disambiguation_data['task_id'] = self.current_task.get('id', 'unknown') if self.current_task and isinstance(self.current_task, dict) else 'unknown'
        disambiguation_data['step'] = self.current_task.get('step_count', 0) if self.current_task and isinstance(self.current_task, dict) else 0
        
        # Schedule broadcast if event loop is running
        self._schedule_broadcast()
   
    def set_user_selection(self, selection: int):
        """Set user selection and save to history"""
        self.user_selection = selection
        
        # Save completed disambiguation to history
        if self.disambiguation_data:
            completed_disambiguation = {
                **self.disambiguation_data,  # Copy all original data
                'user_selection': selection,
                'end_time': datetime.now().isoformat(),
                'selection_method': 'user_choice' if selection else 'auto_timeout',
                'selected_object': None
            }
            
            # Find which object was selected
            if selection and 'candidates' in self.disambiguation_data:
                try:
                    selected_idx = selection - 1  # Convert 1-based to 0-based
                    if 0 <= selected_idx < len(self.disambiguation_data['candidates']):
                        completed_disambiguation['selected_object'] = self.disambiguation_data['candidates'][selected_idx]
                except (IndexError, TypeError):
                    pass
            
            self.disambiguation_history.append(completed_disambiguation)
            
            # Log this as an interaction too
            selected_obj = completed_disambiguation.get('selected_object')
            reasoning = 'Unknown'
            if selected_obj and isinstance(selected_obj, dict):
                reasoning = selected_obj.get('reasoning', 'Unknown')

            self.add_interaction({
                'type': 'disambiguation_complete',
                'action': f"Selected {self.disambiguation_data.get('object_type', 'object')} option {selection}",
                'content': f"Choice: {reasoning}",
                'step': completed_disambiguation.get('step', 0)
            })
        
        self.disambiguation_active = False
        self.disambiguation_data = None
        # Schedule broadcast if event loop is running
        self._schedule_broadcast()
   
    def add_vlm_call(self, vlm_data: Dict):
        """Add VLM call record - enhanced version"""
        analysis_type = vlm_data.get('analysis_type', 'general')
       
        if vlm_data.get('success', True):  # Successful VLM call
            action_text = f"VLM Analysis: {analysis_type}"
            if 'duration' in vlm_data:
                action_text += f" ({vlm_data['duration']}s)"
               
            content_parts = []
            if 'response_preview' in vlm_data:
                content_parts.append(f"Response: {vlm_data['response_preview']}")
            if 'prompt_preview' in vlm_data:
                content_parts.append(f"Prompt: {vlm_data['prompt_preview']}")
            if 'confidence' in vlm_data:
                content_parts.append(f"Confidence: {vlm_data['confidence']}%")
               
            content = " | ".join(content_parts)
           
        else:  # Failed VLM call
            action_text = f"VLM Analysis Failed: {analysis_type}"
            if 'duration' in vlm_data:
                action_text += f" ({vlm_data['duration']}s)"
            content = f"Error: {vlm_data.get('error', 'Unknown error')}"
       
        self.add_interaction({
            'type': 'vlm_call',
            'action': action_text,
            'content': content,
            'image_path': vlm_data.get('image_path', ''),
            'step': vlm_data.get('step', 0),
            'vlm_details': vlm_data
        })