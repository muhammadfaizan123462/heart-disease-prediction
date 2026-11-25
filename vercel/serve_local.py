"""Run the Vercel handler locally for testing.

Usage (from the vercel/ directory, with the project venv):
    ../.venv/bin/python serve_local.py [port]

Then open http://127.0.0.1:8000 in a browser, or curl:
    curl http://127.0.0.1:8000/api/health
    curl -X POST http://127.0.0.1:8000/api/predict \
        -H 'Content-Type: application/json' \
        -d '[57,1,2,124,261,0,0,141,0,0.3,1,0,7]'
"""

from __future__ import annotations

import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "api"))

from index import handler  # noqa: E402


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving on http://127.0.0.1:{port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
