"""
IMPORTS
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
import subprocess
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
AI HELPER
"""


# The full illustrations of borderless and full art cards (ai_art/server.py), when it's set up (python ai_art/setup.py): it runs
# with its own Python, in the background, while Card Conjurer does
def start_ai_helper():
    python = os.path.join(DIRECTORY, "ai_art", ".venv", "Scripts", "python.exe")
    server = os.path.join(DIRECTORY, "ai_art", "server.py")
    if not (os.path.exists(python) and os.path.exists(server)):
        return None
    log = open(os.path.join(DIRECTORY, "ai_art", "server.log"), "w", encoding="utf-8")
    print("AI helper started http://%s:8081 (full illustrations of borderless cards)" % NAME)
    return subprocess.Popen([python, server, "--parent", str(os.getpid())], cwd=DIRECTORY, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

"""
START APP
"""

if __name__ == "__main__":
    # one thread per connection: a connection the browser opens ahead and leaves idle can't hold up the others
    webServer = ThreadingHTTPServer((NAME, PORT), Handler)
    print("Server started http://%s:%s" % (NAME, PORT))
    aiHelper = start_ai_helper()

    try:
        webbrowser.open('http://localhost:8080', new=2)
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    if aiHelper:
        aiHelper.terminate()
    print("Server stopped.")
