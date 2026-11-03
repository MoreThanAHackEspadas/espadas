"""External (non-BAR-ATH) games used to tune and validate settings.

Per the organisers' clarification, every parameter is tuned on external data only. The data
lives outside this repo under MTAH_EXTERNAL_DIR (default: ../external_data next to the repo).
"""

import os

from .. import config


def external_dir():
    return os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
        os.path.dirname(config.REPO_ROOT), "external_data")


def metrica_dir():
    return os.path.join(external_dir(), "metrica")
