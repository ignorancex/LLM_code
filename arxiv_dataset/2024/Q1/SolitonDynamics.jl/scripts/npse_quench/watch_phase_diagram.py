from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import phase_diagram

def exec(path):
  try:
    phase_diagram.fill_phase_diagram() 
  except Exception as e:
    print(e)

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if event.is_directory:
            return
           
        print("="*50)
        print(f"File {event.src_path} has been modified.")
        self.callback(event.src_path)

if __name__ == "__main__":
    exec("")
    phase_diagram.fill_phase_diagram()
    event_handler = FileChangeHandler(exec)
    observer = Observer()
    observer.schedule(event_handler, path='/home/lorenzi/sw/SolitonDynamics.jl/results', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:   
        observer.stop()

    observer.join()