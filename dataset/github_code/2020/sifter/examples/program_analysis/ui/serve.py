"""Starts a server displaying an interactive version of a code analogy."""
import os
import json
import http.server
import socketserver
from ui.lazy_structure_parser import parse_lazy_structure

PORT = 8001

# https://stackoverflow.com/questions/18444395
class RequestHandler(http.server.BaseHTTPRequestHandler):
    """Server handler for code triplet structures."""
    def do_GET(self):
        """Serves static content and parsed structures."""

        if self.path == "/Structure":
            self.send_good_headers("application/json")
            structure = parse_lazy_structure(self.structure)
            self.wfile.write(json.dumps(structure).encode())
        elif self.path.count("/") == 1:
            path = os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
            path += f"/examples/program_analysis/ui{self.path}"
            if self.path == "/":
                path += "index.html"
            try:
                with open(path, "r") as disk_file:
                    data = disk_file.read().encode()
            except OSError:
                self.send_response(404)
            else:
                if ".css" in self.path:
                    self.send_good_headers("text/css")
                else:
                    self.send_good_headers("text/html")
                self.wfile.write(open(path).read().encode())
        else:
            self.send_response(404)

    def send_good_headers(self, content_type):
        """Send a 200 along with the given content_type."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-type", content_type)
        self.end_headers()

class ReuseAddrTCPServer(socketserver.TCPServer):
    """Server allowing to start over an existing port.

    https://stackoverflow.com/questions/15260558
    """
    allow_reuse_address = True

def start_server(structure):
    """Start serving an interactive version of the LazyStructure @structure."""
    RequestHandler.structure = structure
    with ReuseAddrTCPServer(("", PORT), RequestHandler) as httpd:
        print(f"Result available at http://localhost:{PORT}")
        httpd.serve_forever()
