"""Download Metrica Sports' public sample games (external tuning data) outside this repo.

    export MTAH_EXTERNAL_DIR="<workdir>/external_data"
    python tools/fetch_metrica.py                # -> $MTAH_EXTERNAL_DIR/metrica/...

Source: https://github.com/metrica-sports/sample-data, pinned to one commit so every member gets
the same bytes. No formal licence; the README asks to "be responsible with the use of this
data" and to acknowledge the source. Never commit these files (the data guard blocks them).
Already-downloaded files with the right size are skipped.
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import external  # noqa: E402

COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/".format(COMMIT)

# path in the source repo -> size in bytes at COMMIT
FILES = {
    "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv": 139446,
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": 32782073,
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": 32782961,
    "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv": 153897,
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": 29627711,
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": 31890338,
    "data/Sample_Game_3/Sample_Game_3_events.json": 1709486,
    "data/Sample_Game_3/Sample_Game_3_metadata.xml": 81806,
    "data/Sample_Game_3/Sample_Game_3_tracking.txt": 53443115,
    "documentation/events-definitions.pdf": 802916,
}


def main():
    root = external.metrica_dir()
    for rel, size in FILES.items():
        dest = os.path.join(root, rel)
        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            print("ok    {}".format(rel))
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print("get   {}".format(rel))
        tmp = dest + ".part"
        urllib.request.urlretrieve(BASE + rel, tmp)
        got = os.path.getsize(tmp)
        if got != size:
            os.remove(tmp)
            raise IOError("{}: expected {} bytes, got {}".format(rel, size, got))
        os.replace(tmp, dest)
    print("\nMetrica sample data in {}".format(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
