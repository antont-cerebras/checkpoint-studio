# `boto3` — one function, and the four S3 calls the scripts make.
#
# The client is stubbed as a class with named methods rather than boto3's real
# runtime-generated `BaseClient` (whose methods don't exist until botocore reads its JSON
# service model, which is why editors and type checkers can't see them either). Declaring
# them explicitly means a typo in `list_objects_v2` is caught here, and the list doubles as
# the audit of what these scripts do to S3: three reads and a tag read, no writes.
#
# Responses are `dict[str, Any]`: they're deeply nested JSON-ish shapes and the scripts pull
# a few keys out with `.get`, so a fuller model would be guesswork.
#
# (Prose lives in comments, not docstrings: a docstring in a stub is itself a lint —
# PYI021 — because stubs carry types, and these notes are for whoever edits the stub.)

from typing import Any

from botocore.config import Config

class S3Client:
    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = ...,
        ContinuationToken: str = ...,
        MaxKeys: int = ...,
        Delimiter: str = ...,
    ) -> dict[str, Any]: ...
    def head_object(
        self, *, Bucket: str, Key: str, ChecksumMode: str = ...
    ) -> dict[str, Any]: ...
    def get_object(
        self, *, Bucket: str, Key: str, Range: str = ...
    ) -> dict[str, Any]: ...
    def get_object_tagging(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...

def client(
    service_name: str, /, *, config: Config | None = ..., **kwargs: Any
) -> S3Client: ...

__all__ = ["S3Client", "client"]
