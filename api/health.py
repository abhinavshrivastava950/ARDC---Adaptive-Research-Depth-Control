from http.server import BaseHTTPRequestHandler

from _common import respond, service


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        respond(self, service.handle_health())

    def log_message(self, *a):
        pass
