from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import zarr
from upath import UPath
from zarr.storage import FsspecStore, LocalStore, MemoryStore

import spatialdata._store
from spatialdata import SpatialData
from spatialdata._io._utils import _resolve_zarr_store
from spatialdata._store import (
    normalize_path,
    open_read_store,
    open_write_store,
    parquet_fs_and_path,
    path_from_store,
    store_from_group,
)
from spatialdata.testing import assert_spatial_data_objects_are_identical


def test_normalize_path_local_string(tmp_path: Path) -> None:
    result = normalize_path(str(tmp_path / "store.zarr"))
    assert isinstance(result, Path)


def test_normalize_path_remote_string() -> None:
    result = normalize_path("s3://bucket/store.zarr")
    assert isinstance(result, UPath)


def test_normalize_path_storage_options() -> None:
    result = normalize_path("s3://bucket/store.zarr", storage_options={"anon": True})
    assert isinstance(result, UPath)
    assert getattr(result.fs, "anon", None) is True


def test_normalize_path_passthrough_path(tmp_path: Path) -> None:
    p = tmp_path / "store.zarr"
    assert normalize_path(p) is p


def test_normalize_path_passthrough_upath() -> None:
    u = UPath("s3://bucket/store.zarr")
    assert normalize_path(u) is u


def test_open_read_and_write_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "store.zarr"

    with open_write_store(path) as store:
        group = zarr.create_group(store=store, overwrite=True)
        group.attrs["answer"] = 42

    with open_read_store(path) as store:
        group = zarr.open_group(store=store, mode="r")
        assert group.attrs["answer"] == 42


def test_path_from_store_local(tmp_path: Path) -> None:
    """Path derivation from a LocalStore returns the on-disk root as a Path."""
    path = tmp_path / "store.zarr"

    with open_write_store(path) as store:
        zarr.create_group(store=store, overwrite=True)
        assert path_from_store(store) == path


def test_store_from_group_local_subroots(tmp_path: Path) -> None:
    """`store_from_group` returns a LocalStore re-rooted at the group's path."""
    path = tmp_path / "store.zarr"

    with open_write_store(path) as store:
        root = zarr.create_group(store=store, overwrite=True)
        group = root.require_group("images").require_group("image")

        sub = store_from_group(group, read_only=True)
        assert isinstance(sub, LocalStore)
        assert Path(sub.root) == path / "images" / "image"
        assert sub.read_only is True


def test_parquet_fs_and_path_local(tmp_path: Path) -> None:
    """`parquet_fs_and_path` returns an fsspec LocalFileSystem and a joined local path string."""
    from fsspec.implementations.local import LocalFileSystem

    path = tmp_path / "store.zarr"
    with open_write_store(path) as store:
        root = zarr.create_group(store=store, overwrite=True)
        group = root.require_group("points").require_group("p1")

        fs, parquet_path = parquet_fs_and_path(group, "points.parquet")
        assert isinstance(fs, LocalFileSystem)
        assert parquet_path == str(path / "points" / "p1" / "points.parquet")


def test_resolve_zarr_store_returns_existing_zarr_stores_unchanged() -> None:
    """StoreLike inputs must not be wrapped as FsspecStore(fs=store) -- that is only for async filesystems."""
    mem = MemoryStore()
    assert _resolve_zarr_store(mem) is mem
    loc = LocalStore(tempfile.mkdtemp())
    assert _resolve_zarr_store(loc) is loc


def test_resolve_zarr_store_forwards_read_only_local(tmp_path: Path) -> None:
    """`_resolve_zarr_store(..., read_only=True)` must reach the LocalStore constructor."""
    store = _resolve_zarr_store(tmp_path / "store.zarr", read_only=True)
    assert isinstance(store, LocalStore)
    assert store.read_only is True


def test_resolve_zarr_store_forwards_read_only_remote() -> None:
    """`_resolve_zarr_store(..., read_only=True)` must reach the FsspecStore constructor."""
    from fsspec.implementations.memory import MemoryFileSystem

    upath = UPath("memory://ro-remote.zarr", fs=MemoryFileSystem(skip_instance_cache=True))
    store = _resolve_zarr_store(upath, read_only=True)
    assert isinstance(store, FsspecStore)
    assert store.read_only is True


def test_path_from_store_remote() -> None:
    """`path_from_store` on a remote FsspecStore yields a UPath with the original sync fs."""
    import fsspec
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    fs = fsspec.filesystem("memory")
    async_fs = AsyncFileSystemWrapper(fs, asynchronous=True)
    store = FsspecStore(async_fs, path="/some/remote.zarr")

    result = path_from_store(store)
    assert isinstance(result, UPath)
    assert getattr(result.fs, "protocol", None) == "memory"


def test_path_from_store_memory_returns_none() -> None:
    """Stores without a meaningful filesystem path (MemoryStore, custom) return None."""
    assert path_from_store(MemoryStore()) is None


# ---------------------------------------------------------------------------
# Public-API: passing a zarr StoreLike directly to write() / read_zarr().
# This is the headline capability of the refactor -- users hand us a configured
# zarr store (e.g. FsspecStore with embedded credentials) instead of a UPath.
# ---------------------------------------------------------------------------


def test_write_and_read_via_local_store(points: SpatialData, tmp_path: Path) -> None:
    """`write(store)` and `read_zarr(store)` round-trip through a zarr LocalStore.

    Exercises the StoreLike branches in ``write()`` (path_from_store) and
    ``read_zarr()`` (use the store directly), and that ``sdata.path`` is derived
    from the store.
    """
    store_path = tmp_path / "store.zarr"

    write_store = LocalStore(str(store_path))
    points.write(write_store, overwrite=True)
    # sdata.path is derived from the store, as a plain Path
    assert points.path == store_path

    read_store = LocalStore(str(store_path), read_only=True)
    read = SpatialData.read(read_store)
    assert read.path == store_path
    assert_spatial_data_objects_are_identical(points, read)


def test_write_to_memory_store_raises() -> None:
    """A store with no filesystem path (MemoryStore) is rejected with a clear error."""
    sdata = SpatialData()
    with pytest.raises(NotImplementedError, match="does not expose a filesystem path"):
        sdata.write(MemoryStore())


def test_parquet_fs_fallback_is_called_for_a_store_without_a_filesystem() -> None:
    """The fallback supplies the filesystem that `parquet_fs_and_path` cannot derive from the store."""
    root = zarr.create_group(store=MemoryStore(), zarr_format=3)
    group = root.require_group("points").require_group("p1")
    sentinel = object()

    with pytest.raises(ValueError, match="Cannot derive a filesystem"):
        parquet_fs_and_path(group, "points.parquet")

    def fallback(store, resolved_group, child):
        assert isinstance(store, MemoryStore)
        return sentinel, f"{resolved_group.path}/{child}"

    spatialdata._store.parquet_fs_fallback = fallback
    try:
        fs, path = parquet_fs_and_path(group, "points.parquet")
        assert fs is sentinel
        assert path == "points/p1/points.parquet"
    finally:
        spatialdata._store.parquet_fs_fallback = None

    with pytest.raises(ValueError, match="Cannot derive a filesystem"):
        parquet_fs_and_path(group, "points.parquet")


def test_parquet_fs_fallback_is_not_called_for_a_local_store(tmp_path: Path) -> None:
    """A fallback does not replace the filesystem `parquet_fs_and_path` derives for a local store."""

    def fallback(store, group, child):
        raise AssertionError(f"fallback called for {type(store).__name__}")

    spatialdata._store.parquet_fs_fallback = fallback
    try:
        with open_write_store(tmp_path / "store.zarr") as store:
            root = zarr.create_group(store=store, overwrite=True)
            group = root.require_group("points").require_group("p1")
            fs, path = parquet_fs_and_path(group, "points.parquet")
            assert path == str(tmp_path / "store.zarr" / "points" / "p1" / "points.parquet")
    finally:
        spatialdata._store.parquet_fs_fallback = None


def _copy_store(source: LocalStore) -> MemoryStore:
    """A MemoryStore that holds the same keys and bytes as `source`."""

    async def copy() -> MemoryStore:
        memory = MemoryStore()
        prototype = zarr.core.buffer.default_buffer_prototype()
        async for key in source.list():
            await memory.set(key, await source.get(key, prototype=prototype))
        return memory

    return asyncio.run(copy())


def test_read_rasters_from_a_store_without_a_path(images: SpatialData, tmp_path: Path) -> None:
    """The raster reader reads the element group of a `MemoryStore`, not the container root."""
    path = tmp_path / "images.zarr"
    images.write(path)

    read = SpatialData.read(_copy_store(LocalStore(str(path), read_only=True)))
    assert_spatial_data_objects_are_identical(images, read)


def test_read_table_from_a_store_without_a_path(full_sdata: SpatialData, tmp_path: Path) -> None:
    """The table reader receives the table group, not the container root."""
    path = tmp_path / "full.zarr"
    full_sdata.write(path)

    read = SpatialData.read(_copy_store(LocalStore(str(path), read_only=True)), selection=("tables",))
    assert set(read.tables) == set(full_sdata.tables)
    for name, table in read.tables.items():
        assert table.shape == full_sdata.tables[name].shape
