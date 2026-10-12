"""
IMPORTS
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
import os

"""
SETTINGS
"""

NAME = "localhost"
PORT = 8080
DIRECTORY = os.getcwd()

"""
REQUEST HANDLER
"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    # The browser asks again for every file on each load (unchanged files answer "304 Not Modified"), so an updated
    # Card Conjurer shows right away instead of an old cached copy of its scripts
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    # Text files are UTF-8: said so, the browser never reads a script with another encoding
    def guess_type(self, path):
        mimetype = super().guess_type(path)
        if mimetype.startswith("text/") or mimetype in ("application/javascript", "application/json", "image/svg+xml"):
            return mimetype + "; charset=utf-8"
        return mimetype

"""
START APP
"""

if __name__ == "__main__":
    # one thread per connection: a connection the browser opens ahead and leaves idle can't hold up the others
    webServer = ThreadingHTTPServer((NAME, PORT), Handler)
    print("Server started http://%s:%s" % (NAME, PORT))

    try:
        webbrowser.open('http://localhost:8080', new=2)
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
