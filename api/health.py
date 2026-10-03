import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cglc import service  # noqa: E402
from cglc.vercel_api import read_body, respond  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        respond(self, service.handle_health())

    def log_message(self, *args):  # request bodies contain API keys: log nothing
        pass
