from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import plotting_base

def exec(path):
  try:
    plotting_base.plot_axial()
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