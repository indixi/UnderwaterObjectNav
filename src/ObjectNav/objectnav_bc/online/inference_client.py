"""System-Python client for the localhost Conda inference worker."""

import socket
import threading

from .ipc_protocol import receive_message, send_message


class InferenceClient:
    def __init__(self, host, port, connect_timeout_s=1.0):
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.socket = None
        self.lock = threading.Lock()

    @property
    def connected(self):
        return self.socket is not None

    def connect(self):
        connection = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s)
        try:
            connection.settimeout(self.connect_timeout_s)
            send_message(connection, {"type": "HEALTH"})
            response, _ = receive_message(connection)
            if not response.get("ok") or response.get("type") != "READY":
                raise ConnectionError(
                    "inference worker did not report READY")
            connection.settimeout(None)
        except Exception:
            connection.close()
            raise
        self.socket = connection

    def request(self, payload, arrays=None, timeout_s=None):
        with self.lock:
            if self.socket is None:
                raise ConnectionError("inference worker is not connected")
            previous_timeout = self.socket.gettimeout()
            self.socket.settimeout(timeout_s)
            try:
                send_message(self.socket, payload, arrays)
                response, result_arrays = receive_message(self.socket)
            except Exception:
                self.close()
                raise
            finally:
                if self.socket is not None:
                    self.socket.settimeout(previous_timeout)
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "inference worker error"))
            return response, result_arrays

    def close(self):
        connection = self.socket
        self.socket = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
