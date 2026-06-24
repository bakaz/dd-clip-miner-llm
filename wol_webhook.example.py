# =============================================================================
# wol_webhook.example.py — DDTV 录播结束 → Wake-on-LAN 唤醒 GPU 机
# =============================================================================
# 【作用】在 DDTV 录播机本地监听 WebHook；指定主播 RecordingEnd 时向 GPU 机发 WOL 魔术包。
#
# 【整体流程】
#   DDTV 录播结束 → POST /webhook (cmd=RecordingEnd)
#        → 校验 UID == TARGET_UID
#        → send_wol(GPU_MAC) 广播魔术包
#        → GPU 机唤醒 → Windows 计划任务跑 batch-run + cut_copy
#
# 部署步骤:
#   1. copy wol_webhook.example.py wol_webhook.py
#   2. 修改下方 GPU_MAC / TARGET_UID 等
#   3. python wol_webhook.py          # 前台
#      python wol_webhook.py -t         # 测试：发一次 WOL 后退出
#      pythonw wol_webhook.py           # 后台无窗口
#
# DDTV 设置（录播机）:
#   WebHookSwitch = true
#   WebHookAddress = http://127.0.0.1:29000/webhook
# =============================================================================

from __future__ import annotations

import argparse
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- 本地配置（复制为 wol_webhook.py 后按环境修改）---------------------------
GPU_MAC = "AA-BB-CC-DD-EE-FF"   # GPU 机网卡 MAC（格式 AA-BB-CC-DD-EE-FF 或 AA:BB:...）；WOL 包目标
GPU_IP = "192.168.1.100"        # GPU 机 IP，仅打印日志用；WOL 实际发往 255.255.255.255:9
TARGET_UID = 123456             # DDTV 主播 UID；仅 data.UID 与此相等且 cmd=RecordingEnd 时发 WOL
TARGET_ROOM = 92450             # DDTV 房间号，仅启动日志显示，不参与过滤逻辑
TARGET_NAME = "StreamerName"    # 主播显示名，仅启动日志显示
LISTEN_PORT = 29000             # 本机 HTTP 监听端口；DDTV WebHookAddress 须指向此端口 /webhook
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
