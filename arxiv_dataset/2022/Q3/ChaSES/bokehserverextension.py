# adapted from https://github.com/binder-examples/bokeh

from subprocess import Popen

def load_jupyter_server_extension(nbapp):
    """serve the bokeh-app directory with bokeh server"""
    Popen(["bokeh", "serve", "app", "--allow-websocket-origin=*"])
