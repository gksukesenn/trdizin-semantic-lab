"""Final pipeline use-case entry points.

CLI modules delegate here conceptually; command implementations remain separate so
the numbered scripts stay thin and each long-running stage is independently debugged.
"""

from .commands.build_embeddings import main as build_embeddings
from .commands.build_final_report import main as build_final_report
from .commands.collect_articles import main as collect_articles
from .commands.discover_topics import main as discover_topics
from .commands.validate_dataset import main as validate_dataset

__all__ = ["collect_articles", "validate_dataset", "build_embeddings", "discover_topics", "build_final_report"]
