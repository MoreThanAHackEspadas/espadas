"""The tracking cache (atd.py) is an optimisation only: cached loads equal uncached ones.

Made-up bundles only (tests/conftest.py); the cache lives in tmp_path, never in the repository.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import atd, config  # noqa: E402
from conftest import write_bundle  # noqa: E402

RAW = os.path.join("tactical", "BAR-ATH_tactical_atd.txt")


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    path = str(tmp_path / "cache")
    monkeypatch.setattr(config, "CACHE_DIR", path)
    return path


def same(a, b):
    return all(np.array_equal(a[k], b[k], equal_nan=True) and a[k].dtype == b[k].dtype
               for k in atd.CACHED)


def test_same_name_size_and_mtime_with_different_content_never_share_an_entry(tmp_path,
                                                                             cache_dir):
    a = write_bundle(str(tmp_path / "a"), ball_x_offset=0.5)
    b = write_bundle(str(tmp_path / "b"), ball_x_offset=0.9)
    raw_a, raw_b = os.path.join(a, RAW), os.path.join(b, RAW)
    assert os.path.getsize(raw_a) == os.path.getsize(raw_b)
    with open(raw_a, "rb") as x, open(raw_b, "rb") as y:
        assert x.read() != y.read()
    stat = os.stat(raw_a)
    os.utime(raw_b, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    cached_a = atd.load_tracking(a)               # parses and writes a's entry
    cached_b = atd.load_tracking(b)               # must not be served a's arrays
    fresh_a = atd.load_tracking(a, use_cache=False)
    fresh_b = atd.load_tracking(b, use_cache=False)
    assert not np.array_equal(fresh_a["ball"], fresh_b["ball"], equal_nan=True)
    assert same(cached_a, fresh_a) and same(cached_b, fresh_b)
    assert len(os.listdir(cache_dir)) == 2
    assert same(atd.load_tracking(b), fresh_b)    # a cache hit, still b's arrays


def test_a_cache_entry_with_the_wrong_shapes_is_reparsed(tmp_path, cache_dir):
    bundle = write_bundle(str(tmp_path / "bundle"))
    fresh = atd.load_tracking(bundle, use_cache=False)
    atd.load_tracking(bundle)
    [entry] = os.listdir(cache_dir)
    bad = {k: np.zeros((10,) + fresh[k].shape[1:], dtype=fresh[k].dtype) for k in atd.CACHED}
    with open(os.path.join(cache_dir, entry), "wb") as handle:
        np.savez(handle, **bad)
    assert same(atd.load_tracking(bundle), fresh)
    assert same(atd.load_tracking(bundle), fresh)         # and the entry was rewritten


def test_an_unreadable_cache_entry_is_reparsed(tmp_path, cache_dir):
    bundle = write_bundle(str(tmp_path / "bundle"))
    fresh = atd.load_tracking(bundle, use_cache=False)
    atd.load_tracking(bundle)
    [entry] = os.listdir(cache_dir)
    with open(os.path.join(cache_dir, entry), "wb") as handle:
        handle.write(b"not an npz file")
    assert same(atd.load_tracking(bundle), fresh)


def test_build_with_cache_equals_build_without(build, tmp_path):
    """run.main end to end: no cache, cache written, cache read -> the same bytes."""
    paths = [build("--no-cache", out="clean.csv"), build(out="writes_cache.csv")]
    assert os.listdir(str(tmp_path / "cache"))
    paths.append(build(out="reads_cache.csv"))
    contents = []
    for path in paths:
        with open(path, "rb") as handle:
            contents.append(handle.read())
    assert contents[0] == contents[1] == contents[2]
