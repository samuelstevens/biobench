import hashlib

from . import download as kd


def test_generate_part_files():
    out = kd.generate_part_files("giraffes", "aa", "ab")
    assert out == [
        "dataset/image/giraffes_part_aa",
        "dataset/image/giraffes_part_ab",
    ]


def test_all_files_for_animals_includes_static_and_md5():
    animals = ("giraffes",)
    files = kd.all_files_for_animals(animals)
    # static + md5 + first part file sanity
    assert "README.txt" in files
    assert "dataset/image/giraffes_md5.txt" in files
    assert "dataset/image/giraffes_part_aa" in files
    # no duplicates
    assert len(files) == len(set(files))


def test_md5_file(tmp_path):
    p = tmp_path / "blob.bin"
    data = b"abc" * 123
    p.write_bytes(data)
    expect = hashlib.md5(data).hexdigest()
    assert kd.md5_file(p) == expect


def test_stream_download_stub(tmp_path, monkeypatch):
    # prepare fake response
    payload = b"hello world"

    class FakeResp:
        headers = {"content-length": str(len(payload))}

        def iter_content(self, chunk_size):
            yield payload

        def raise_for_status(self): ...

        # context-manager methods
        def __enter__(self):
            return self

        def __exit__(self, *exc): ...

    monkeypatch.setattr(kd.requests, "get", lambda *a, **kw: FakeResp())
    # run
    dst = tmp_path / "file.bin"
    kd.stream_download("http://foo.bar", dst, 16)
    assert dst.read_bytes() == payload
