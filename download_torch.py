"""下载 torch cu126 wheel 文件（绕过 pip 的代理问题）。

用 requests 库直接下载，支持断点续传与重试。
下载完成后用 pip 从本地文件安装。
"""
import os
import sys
import time
import urllib.request

# 1. 清理 WSL 泄漏的代理环境变量
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
          "all_proxy", "ALL_PROXY", "ftp_proxy", "FTP_PROXY"):
    os.environ.pop(k, None)

# 2. 从 Windows 注册表读出真实代理
_win_proxies = urllib.request.getproxies()
_proxy = (_win_proxies.get("https") or _win_proxies.get("http") or "").replace("https://", "http://")
if _proxy:
    os.environ["http_proxy"] = _proxy
    os.environ["https_proxy"] = _proxy
    print("Using proxy from Windows registry:", _proxy)
else:
    print("No proxy found in registry, trying direct connection")

import requests

WHEEL_URL = "https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp310-cp310-win_amd64.whl"
WHEEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch-2.9.1+cu128-cp310-cp310-win_amd64.whl")

CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_RETRIES = 20


def download():
    # 检查已下载的大小（断点续传）
    downloaded = 0
    if os.path.exists(WHEEL_FILE):
        downloaded = os.path.getsize(WHEEL_FILE)
        print("Resuming from {} ({:.2f} GB)".format(WHEEL_FILE, downloaded / 1024**3))

    for attempt in range(1, MAX_RETRIES + 1):
        headers = {}
        if downloaded > 0:
            headers["Range"] = "bytes={}-".format(downloaded)

        try:
            print("Attempt {}/{}: downloading from offset {}...".format(attempt, MAX_RETRIES, downloaded))
            r = requests.get(WHEEL_URL, headers=headers, stream=True, timeout=60, verify=False)
            total = int(r.headers.get("content-length", 0))

            # 如果服务器返回 200 而非 206，说明不支持断点续传，从头开始
            if downloaded > 0 and r.status_code == 200:
                print("Server doesn't support resume, restarting from 0")
                downloaded = 0
                mode = "wb"
            elif downloaded > 0 and r.status_code == 206:
                total += downloaded
                mode = "ab"
            else:
                mode = "wb"

            print("Total size: {:.2f} GB, status: {}".format(total / 1024**3, r.status_code))

            with open(WHEEL_FILE, mode) as f:
                last_print = time.time()
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_print > 3:
                            pct = (downloaded / total * 100) if total else 0
                            print("  {:.2f} / {:.2f} GB ({:.1f}%)".format(
                                downloaded / 1024**3, total / 1024**3, pct))
                            last_print = now

            print("Download complete: {:.2f} GB".format(os.path.getsize(WHEEL_FILE) / 1024**3))
            return True

        except Exception as e:
            print("Attempt {} failed: {}: {}".format(attempt, type(e).__name__, str(e)[:150]))
            if os.path.exists(WHEEL_FILE):
                downloaded = os.path.getsize(WHEEL_FILE)
                print("  Partial download: {:.2f} GB, will resume".format(downloaded / 1024**3))
            if attempt >= MAX_RETRIES:
                print("All {} attempts exhausted.".format(MAX_RETRIES))
                return False
            backoff = min(5 * attempt, 30)
            print("  Retrying in {}s...".format(backoff))
            time.sleep(backoff)

    return False


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print("=" * 60)
    print("Downloading torch 2.9.1+cu126 for RTX 5070 Ti (sm_120)")
    print("URL:", WHEEL_URL)
    print("Save to:", WHEEL_FILE)
    print("=" * 60)

    if download():
        print("\n✓ Download successful!")
        print("Now installing with pip...")
        print("Run: venv\\Scripts\\python.exe -m pip install --force-reinstall --no-deps \"{}\"".format(WHEEL_FILE))
    else:
        print("\n✗ Download failed.")
        sys.exit(1)
