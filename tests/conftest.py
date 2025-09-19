"""
Global pytest configuration for this repo.

Purpose: Detect and prevent accidental network access during tests.

Behavior:
- By default, any attempt to open a network socket to a non-local address raises
  NetworkAccessError with details about the destination and suggesting how to mock.
- Local addresses (127.0.0.1, ::1, localhost) are allowed.
- Set environment variable ALLOW_NET_TESTS=1 to disable this guard (e.g., for
  explicit integration tests).
"""

from __future__ import annotations

import os
import socket as _socket
from typing import Any

import pytest


class NetworkAccessError(RuntimeError):
    pass


def _is_local_address(address: tuple[str, int]) -> bool:
    host, _port = address
    return host in {"127.0.0.1", "localhost", "::1"}


@pytest.fixture(autouse=True, scope="session")
def _block_network_access():
    if os.getenv("ALLOW_NET_TESTS"):
        return

    original_socket = _socket.socket

    class GuardedSocket(original_socket):
        def connect(self, address: Any) -> None:
            try:
                addr = tuple(address)
            except Exception:
                addr = (str(address), 0)

            if isinstance(addr, tuple) and len(addr) >= 2 and _is_local_address((addr[0], addr[1])):
                return super().connect(address)

            raise NetworkAccessError(
                f"Network access blocked during tests. Attempted to connect to {address}.\n"
                "Mock external calls (e.g., patch httpx/requests or seek.tools) or set "
                "ALLOW_NET_TESTS=1 to explicitly allow network in this run."
            )

    _socket.socket = GuardedSocket

    # Ensure common libraries don’t proxy around the socket-level guard
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("HTTP_PROXY", "")
    os.environ.setdefault("HTTPS_PROXY", "")

    try:
        yield
    finally:
        _socket.socket = original_socket
