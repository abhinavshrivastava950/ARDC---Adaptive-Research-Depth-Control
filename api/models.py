import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cglc import service  # noqa: E402
from cglc.vercel_api import read_body, respond  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        respond(self, service.handle_models(read_body(self)))

    def log_message(self, *args):  # request bodies contain API keys: log nothing
        pass
