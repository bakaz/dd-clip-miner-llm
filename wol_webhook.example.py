# wol_webhook.example.py — DDTV WebHook listener template (WOL on RecordingEnd)
#
# Setup:
#   1. Copy to wol_webhook.py (gitignored local file):
#        copy wol_webhook.example.py wol_webhook.py
#   2. Edit GPU_MAC / GPU_IP / TARGET_UID / TARGET_ROOM / TARGET_NAME in wol_webhook.py
#   3. Run:
#        python wol_webhook.py
#      Test mode (send one WOL packet and exit):
#        python wol_webhook.py -t
#      Background:
#        pythonw wol_webhook.py
#
# DDTV settings:
#   WebHookSwitch = true
#   WebHookAddress = http://127.0.0.1:29000/webhook

from __future__ import annotations

import argparse
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- Local config (replace in wol_webhook.py) --------------------------------
GPU_MAC = "AA-BB-CC-DD-EE-FF"
GPU_IP = "192.168.1.100"
TARGET_UID = 123456
TARGET_ROOM = 92450
TARGET_NAME = "StreamerName"
LISTEN_PORT = 29000
# -----------------------------------------------------------------------------


def send_wol(mac: str) -> bytes:
    """Send a Wake-on-LAN magic packet, returning the built packet bytes."""
    mac_hex = mac.replace(":", "").replace("-", "").replace(".", "")
    if len(mac_hex) != 12:
        raise ValueError(f"Invalid MAC address: {mac!r}, expected 12 hex chars")
    try:
        mac_bytes = bytes.fromhex(mac_hex)
    except ValueError as exc:
        raise ValueError(f"Invalid MAC address characters: {mac!r}") from exc

    packet = b"\xff" * 6 + mac_bytes * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(packet, ("255.255.255.255", 9))
    return packet


class WebHookHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            cmd = data.get("cmd", "")
            msg = data.get("message", "")
            room = data.get("data") or {}
            uid = room.get("UID", 0)
            name = room.get("name", "?")
            room_id = room.get("RoomId", "?")

            print(f"[{cmd}] {name}({room_id}) | {msg}")

            if uid == TARGET_UID and cmd == "RecordingEnd":
                print(f"[WOL] RecordingEnd -> wake GPU ({GPU_MAC} / {GPU_IP})")
                send_wol(GPU_MAC)
                print("[WOL] Done.")
        except Exception as exc:
            print(f"[Error] {exc}")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DDTV WebHook listener that sends a WOL packet on RecordingEnd"
    )
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="Test mode: build and send one WOL packet, then exit"
    )
    args = parser.parse_args()

    if args.test:
        print("[Test Mode] Sending WOL test packet")
        print(f"  target MAC: {GPU_MAC}")
        print(f"  target IP:  {GPU_IP} (for reference only)")
        try:
            packet = send_wol(GPU_MAC)
            print(f"  packet size: {len(packet)} bytes")
            print(f"  packet hex:  {packet.hex()}")
            print("[Test Mode] WOL test packet sent")
        except Exception as exc:
            print(f"[Test Mode Error] {exc}")
            sys.exit(1)
        return

    print("DDTV WebHook Listener")
    print(f"  listen: http://0.0.0.0:{LISTEN_PORT}/webhook")
    print(f"  target: {TARGET_NAME} (UID={TARGET_UID}, RoomId={TARGET_ROOM})")
    print("  event:  RecordingEnd")
    print(f"  gpu:    {GPU_MAC} ({GPU_IP})")
    print()

    server = HTTPServer(("0.0.0.0", LISTEN_PORT), WebHookHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
