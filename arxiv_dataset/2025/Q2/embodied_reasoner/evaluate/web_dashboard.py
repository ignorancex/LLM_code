#!/usr/bin/env python3
"""
New modular web dashboard - uses separate files for better maintainability
"""

import time
from web_ui import start_dashboard_server, log_task_start


if __name__ == "__main__":
    # import unit test
    print("Starting Modular Web Dashboard...")
    start_dashboard_server(port=8888, auto_open=True)
   
    test_task = {
        'identity': 'test_1',
        'taskquery': 'Find the CreditCard in the room',
        'scene': 'FloorPlan1',
        'max_steps': 20
    }
   
    log_task_start(test_task)
   
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nModular Dashboard server stopped")