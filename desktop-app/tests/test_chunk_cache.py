from marginum.gui.chunk_cache import CACHE_PREFIX, ChunkCache


def test_remove_leftovers_deletes_only_cache_folders(tmp_path):
    caches = [tmp_path / f"{CACHE_PREFIX}one", tmp_path / f"{CACHE_PREFIX}two"]
    other = tmp_path / "something-else"
    for folder in (*caches, other):
        folder.mkdir()
        (folder / "chunk.npy").write_bytes(b"x")

    removed = ChunkCache.remove_leftovers(tmp_path)

    assert removed == 2
    assert not any(folder.exists() for folder in caches)
    assert other.exists()
