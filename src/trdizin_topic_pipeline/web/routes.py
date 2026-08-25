"""Demo HTTP route handlers; endpoint contracts live here."""
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from ..services.search_service import SearchService

ROOT = Path(__file__).resolve().parents[3]
WEB_ROOT = ROOT / "web" / "demo"
FIGURES_ROOT = ROOT / "outputs" / "final_50k" / "figures"
APP: Optional[SearchService] = None

def configure_app(application: SearchService) -> None:
    global APP
    APP = application

class DemoHandler(
    BaseHTTPRequestHandler
):
    server_version = (
        "TRDizinSemanticExplorer/1.0"
    )

    def send_json(
        self,
        value: Any,
        status: int = 200,
    ) -> None:
        data = json.dumps(
            value,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(data)),
        )

        self.send_header(
            "Cache-Control",
            "no-store",
        )

        self.end_headers()
        self.wfile.write(data)

    def send_file(
        self,
        path: Path,
    ) -> None:
        if not path.exists():
            self.send_error(404)
            return

        data = path.read_bytes()

        mime_type = (
            mimetypes.guess_type(
                str(path)
            )[0]
            or "application/octet-stream"
        )

        self.send_response(200)

        self.send_header(
            "Content-Type",
            mime_type,
        )

        self.send_header(
            "Content-Length",
            str(len(data)),
        )

        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(
            self.path
        )

        path = unquote(
            parsed.path
        )

        if path == "/":
            self.send_file(
                WEB_ROOT
                / "index.html"
            )
            return

        if path.startswith("/assets/"):
            relative = Path(path[len("/assets/"):])
            if ".." in relative.parts:
                self.send_error(404)
                return
            self.send_file(WEB_ROOT / "assets" / relative)
            return

        if path == "/api/status":
            try:
                assert APP is not None

                self.send_json(
                    APP.status()
                )

            except Exception as error:
                self.send_json(
                    {
                        "error": str(error),
                    },
                    status=500,
                )

            return

        if path.startswith(
            "/figures/"
        ):
            filename = Path(
                path
            ).name

            allowed = {
                "cluster_size_distribution.png",
                "umap_2d_clusters.png",
                "umap_2d_direct_fallback.png",
                "parameter_comparison.png",
            }

            if filename not in allowed:
                self.send_error(404)
                return

            self.send_file(
                FIGURES_ROOT
                / filename
            )

            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(
            self.path
        )

        if parsed.path != "/api/search":
            self.send_error(404)
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            body = self.rfile.read(
                content_length
            )

            request = json.loads(
                body.decode(
                    "utf-8"
                )
            )

            assert APP is not None

            result = APP.search(
                query=str(
                    request.get(
                        "query",
                        "",
                    )
                ),
                mode=str(
                    request.get(
                        "mode",
                        "semantic",
                    )
                ),
                limit=int(
                    request.get(
                        "limit",
                        10,
                    )
                ),
                year_from=(
                    int(
                        request[
                            "year_from"
                        ]
                    )
                    if request.get(
                        "year_from"
                    )
                    not in (
                        None,
                        "",
                    )
                    else None
                ),
                year_to=(
                    int(
                        request[
                            "year_to"
                        ]
                    )
                    if request.get(
                        "year_to"
                    )
                    not in (
                        None,
                        "",
                    )
                    else None
                ),
                database=(
                    str(
                        request.get(
                            "database"
                        )
                    )
                    if request.get(
                        "database"
                    )
                    else None
                ),
            )

            self.send_json(
                result
            )

        except ValueError as error:
            self.send_json(
                {
                    "error": str(error),
                },
                status=400,
            )

        except Exception as error:
            self.send_json(
                {
                    "error": str(error),
                },
                status=500,
            )

    def log_message(
        self,
        format: str,
        *args: Any
    ) -> None:
        sys.stdout.write(
            "[HTTP] "
            + format % args
            + "\n"
        )
