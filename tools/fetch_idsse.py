"""Download the DFL / Sportec "IDSSE" open dataset (external data) outside this repo.

    export MTAH_EXTERNAL_DIR="<workdir>/external_data"
    python tools/fetch_idsse.py                  # -> $MTAH_EXTERNAL_DIR/idsse/..., about 2.6 GB

Source: Bassek, M., Rein, R., Weber, H., Memmert, D. "An integrated dataset of spatiotemporal
and event data in elite soccer", Scientific Data (2025); figshare DOI
10.6084/m9.figshare.28196177, version 1 (published 2 Feb 2025). Seven official matches of the
German Bundesliga (COM-000001, 2 matches) and 2. Bundesliga (COM-000002, 5 matches): match
information, events and observed positions (25 Hz), as DFL XML.

Licence: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/), as stated on the figshare
record. We credit the authors and DFL / Sportec Solutions (README "External data", tech doc
section 3) and do not redistribute the files: they stay outside the repository and the data
guard blocks them. Pinned to version 1; every file is checked against figshare's MD5.
Already-downloaded files with the right MD5 are skipped.
"""

import concurrent.futures
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import external  # noqa: E402

ARTICLE, VERSION = 28196177, 1
DOWNLOAD = "https://ndownloader.figshare.com/files/{}"
WORKERS = 8

# file name -> (figshare file id, size in bytes, md5) at version 1
FILES = {
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml": (51643472, 11722, "37b0f85588c0acfb995fe633c7ac2e0d"),
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml": (51643475, 11930, "bc5d5b21dbcf36aa30f4cd0428309f6b"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml": (51643478, 11745, "b9710781d400dd812f62b0a6e3d7bce8"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml": (51643490, 11823, "5e2d580d3d8ed97e32959de90d04992b"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml": (51643487, 11922, "65a4568e4f374af94d3c1015cbc7c4f5"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml": (51643484, 11932, "5979209af316a914ce57347d0633df6c"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml": (51643481, 11952, "7b021e577fca917a63111d55cc29ae5b"),
    "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml": (51643493, 776578, "2d88e180a1f63c7e4157feb980fc7255"),
    "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml": (51643496, 612114, "83e8ed7ea8e8ebbbdc61f61ac3534008"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml": (51643499, 635747, "ab66e51512f39d06f49f27dd3b4c2f40"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml": (51643502, 683018, "64b1be01d6ff6848168efebd56f344b9"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml": (51643505, 698898, "f68482441d38723d69f244f896482803"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml": (51643508, 732186, "b7e9d82af4034539d802319be6c57776"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml": (51643511, 681050, "c13bc5efa5eca4a022cc678210b87d5f"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml": (51643514, 418524833, "e4273a2adf5dbd9d4ec42def2f01905c"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml": (51643517, 391722803, "bacb2f609c1d1b0bfc906f6b9d4109f2"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml": (51643520, 348945715, "3efff25a86a61768e7d253b4bba137a3"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml": (51643523, 362567859, "4e1063e0a0ae499920cf68ac1aa3ab4e"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml": (51643526, 371688499, "e56ff1336a334f8da51110fbe7cab764"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml": (51643529, 356219423, "9105bad987191d3c3c362c1fce9a4fcd"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml": (51643532, 372934140, "e060bb7d7a5a3bfbd15bbf9bc631024d"),
}


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root, name):
    file_id, size, digest = FILES[name]
    dest = os.path.join(root, name)
    if os.path.isfile(dest) and os.path.getsize(dest) == size and md5(dest) == digest:
        return "ok    {}".format(name)
    tmp = dest + ".part"
    urllib.request.urlretrieve(DOWNLOAD.format(file_id), tmp)
    if os.path.getsize(tmp) != size or md5(tmp) != digest:
        os.remove(tmp)
        raise IOError("{}: size or MD5 does not match figshare v{}".format(name, VERSION))
    os.replace(tmp, dest)
    return "got   {} ({:.0f} MB)".format(name, size / 1e6)


def main():
    root = external.idsse_dir()
    os.makedirs(root, exist_ok=True)
    names = sorted(FILES, key=lambda n: FILES[n][1])
    # figshare serves one file slowly (~0.5 MB/s); several at once fill the line
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for line in pool.map(lambda n: fetch(root, n), names):
            print(line, flush=True)
    print("\nIDSSE (figshare {} v{}) in {}".format(ARTICLE, VERSION, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
