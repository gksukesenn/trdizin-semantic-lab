#!/usr/bin/env python3
"""TR Dizin demo sunucusunun composition root ve yaşam döngüsü."""
import argparse
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from ..services.search_service import SearchService
from .routes import DemoHandler, configure_app

ROOT = Path(__file__).resolve().parents[3]
APP: Optional[SearchService] = None

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/final_50k.json",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    global APP

    args = parse_arguments()

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():
        config_path = (
            ROOT
            / config_path
        )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    APP = SearchService(
        config=config,
        allow_cpu=args.allow_cpu,
    )
    configure_app(APP)

    server = ThreadingHTTPServer(
        (
            args.host,
            args.port,
        ),
        DemoHandler,
    )

    print(
        "\nDemo adresi: http://%s:%d"
        % (
            args.host,
            args.port,
        )
    )

    print(
        "Durdurmak için Ctrl+C.\n"
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nSunucu kapatılıyor...")

    finally:
        server.server_close()

        if APP is not None:
            APP.store.close()
