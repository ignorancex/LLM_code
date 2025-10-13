#!/usr/bin/env python3
import os
import json
import time
import threading
import webbrowser
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .monitor import AgentMonitor


# Global monitor instance
monitor = AgentMonitor()

# FastAPI Application
app = FastAPI(title="Embodied Agent Monitor", description="Real-time monitoring dashboard")

# Get the directory of this file to locate static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML"""
    template_path = os.path.join(BASE_DIR, "templates", "dashboard.html")
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/api/state")
async def get_current_state():
    """Provide current state REST API for polling"""
    state_data = {
        'type': 'state_update',
        'current_task': monitor.current_task,
        'task_history': list(monitor.task_history),
        'interaction_log': list(monitor.interaction_log),
        'task_stats': monitor.task_stats,
        'disambiguation_active': monitor.disambiguation_active,
        'disambiguation_data': monitor.disambiguation_data,
        'timestamp': datetime.now().isoformat()
    }
    return state_data


@app.get("/image/{image_path:path}")
async def serve_image(image_path: str):
    """Serve image files"""
    try:
        # Security check to prevent path traversal
        if ".." in image_path or image_path.startswith("/"):
            raise HTTPException(status_code=403, detail="Access denied")
           
        full_path = os.path.join("/home/jiajunliu/embodied_reasoner", image_path)
       
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return FileResponse(full_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await monitor.connect(websocket)
    try:
        while True:
            # Receive client message
            data = await websocket.receive_text()
            message = json.loads(data)
           
            # Process user selection
            if message.get('type') == 'user_selection':
                monitor.set_user_selection(message.get('selection'))
               
    except WebSocketDisconnect:
        monitor.disconnect(websocket)


def start_dashboard_server(port: int = 8888, auto_open: bool = True):
    """Start the dashboard server"""
    def run_server():
        try:
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
        except Exception as e:
            print(f"Dashboard server failed to start: {e}")
   
    # Start the server thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
   
    # Wait for the server to start
    time.sleep(2)
   
    url = f"http://localhost:{port}"
    print(f"Dashboard available at: {url}")
   
    if auto_open:
        try:
            webbrowser.open(url)
            print("Browser opened automatically")
        except:
            print("Could not auto-open browser, please visit the URL manually")
   
    return server_thread


# Convenient global functions for RocAgent to call
def log_task_start(task_data: dict):
    """Log task start"""
    monitor.start_task(task_data)


def log_interaction(interaction_data: dict):
    """Log interaction"""
    monitor.add_interaction(interaction_data)


def log_task_complete(success: bool, result_data: dict = None):
    """Log task completion"""
    monitor.complete_task(success, result_data)


def log_vlm_call(vlm_data: dict):
    """Log VLM call"""
    monitor.add_vlm_call(vlm_data)


def start_disambiguation_web(disambiguation_data: dict, timeout: int = 30) -> Optional[int]:
    """Start web-based disambiguation interface"""
    try:
        monitor.start_disambiguation(disambiguation_data)
       
        # Use provided timeout or default to 30
        start_time = time.time()
       
        while monitor.user_selection is None:
            time.sleep(0.5)
            if time.time() - start_time > timeout:
                print(f"Web disambiguation timed out ({timeout}s), triggering VLM analysis")
                # Signal timeout to RocAgent for VLM analysis
                # Use special value -1 to indicate timeout (need VLM analysis)
                monitor.set_user_selection(-1)
                print(f"Web UI timeout: Returning control to RocAgent for VLM analysis")
                break
       
        result = monitor.user_selection
        if result == -1:
            print(f"Web disambiguation completed: TIMEOUT (will trigger VLM analysis)")
        else:
            print(f"Web disambiguation completed, selection: Option {result}")
        return result
       
    except Exception as e:
        print(f"Web disambiguation failed: {e}")
        return 1