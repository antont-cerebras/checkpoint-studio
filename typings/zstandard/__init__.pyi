# `zstandard` — the one-shot `decompress` used to inflate compressed S3 objects.

def decompress(data: bytes, max_output_size: int = ...) -> bytes: ...

__all__ = ["decompress"]
