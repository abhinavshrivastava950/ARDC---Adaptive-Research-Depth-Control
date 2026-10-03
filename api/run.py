from http.server import BaseHTTPRequestHandler

from _common import read_body, respond, service


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        respond(self, service.handle_run(read_body(self)))

    def log_message(self, *a):  # request bodies contain API keys: log nothing
        pass
