#!/usr/bin/env python3
"""
简单的 HTTP 服务器，用于提供前端静态文件
解决直接打开 HTML 文件时的 CORS 问题
"""
import http.server
import socketserver
import os
from pathlib import Path

# 获取前端目录
FRONTEND_DIR = Path(__file__).parent / "frontend"
PORT = 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # 添加 CORS 头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    if not FRONTEND_DIR.exists():
        print(f"❌ 错误: 前端目录不存在: {FRONTEND_DIR}")
        return
    
    if not (FRONTEND_DIR / "index.html").exists():
        print(f"❌ 错误: 找不到 index.html 文件: {FRONTEND_DIR / 'index.html'}")
        return
    
    try:
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🌐 前端服务器已启动！")
        print(f"📂 服务目录: {FRONTEND_DIR}")
        print(f"🔗 访问地址: http://localhost:{PORT}")
        print(f"💡 请确保 API 服务运行在 http://localhost:8000")
        print(f"\n按 Ctrl+C 停止服务器\n")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ 错误: 端口 {PORT} 已被占用，请先关闭占用该端口的程序")
        else:
            print(f"❌ 错误: {e}")
        except KeyboardInterrupt:
            print("\n\n👋 服务器已停止")

if __name__ == "__main__":
    main()


