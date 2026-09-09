"""Small length-prefixed JSON + NumPy protocol for localhost inference."""

import json
import struct

import numpy as np


PROTOCOL_VERSION = 1
_LENGTH = struct.Struct("!I")
MAX_HEADER_BYTES = 1024 * 1024
MAX_ARRAY_BYTES = 64 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


def _recv_exact(sock, size):
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("inference socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock, payload, arrays=None):
    arrays = arrays or {}
    document = dict(payload)
    document["protocol_version"] = PROTOCOL_VERSION
    metadata = []
    buffers = []
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        if array.nbytes > MAX_ARRAY_BYTES:
            raise ProtocolError(
                "array %r is too large: %d bytes" % (name, array.nbytes))
        metadata.append(
            {
                "name": str(name),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": int(array.nbytes),
            }
        )
        buffers.append(memoryview(array).cast("B"))
    document["arrays"] = metadata
    header = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ProtocolError("message header is too large")
    sock.sendall(_LENGTH.pack(len(header)))
    sock.sendall(header)
    for buffer in buffers:
        sock.sendall(buffer)


def receive_message(sock):
    header_size = _LENGTH.unpack(_recv_exact(sock, _LENGTH.size))[0]
    if header_size <= 0 or header_size > MAX_HEADER_BYTES:
        raise ProtocolError("invalid message header size: %d" % header_size)
    document = json.loads(_recv_exact(sock, header_size).decode("utf-8"))
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            "protocol version mismatch: %r" % document.get("protocol_version"))
    arrays = {}
    for metadata in document.pop("arrays", []):
        name = str(metadata["name"])
        dtype = np.dtype(metadata["dtype"])
        shape = tuple(int(value) for value in metadata["shape"])
        nbytes = int(metadata["nbytes"])
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != expected or nbytes < 0 or nbytes > MAX_ARRAY_BYTES:
            raise ProtocolError("invalid array metadata for %r" % name)
        value = np.frombuffer(_recv_exact(sock, nbytes), dtype=dtype).copy()
        arrays[name] = value.reshape(shape)
    return document, arrays
