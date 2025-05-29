import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
from PIL import Image
import numpy as np
import hashlib

# ─── 1. 设置代理和跳过列表 ─────────────────────────────────
os.environ.setdefault("HTTP_PROXY",  "http://your-real-proxy:8080")
os.environ.setdefault("HTTPS_PROXY", "http://your-real-proxy:8080")
os.environ.setdefault("NO_PROXY",      "picsum.photos")

# ─── 2. 带重试的 Session ───────────────────────────────────
session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

def fetch_random_image(h: int, w: int) -> Image.Image:
    """直接从 picsum.photos 拉取随机图并返回 PIL.Image"""
    resp = session.get(f"https://picsum.photos/{h}/{w}", timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))

def image_hash(img: Image.Image) -> str:
    """像素级 MD5 哈希，用于去重"""
    arr = np.array(img)
    return hashlib.md5(arr.tobytes()).hexdigest()

def generate_and_save(n=20, height=1000, width=1000, base_dir="images"):
    # 在 base_dir 下创建子目录 images_{height}x{width}
    outdir = os.path.join(base_dir, f"images_{height}x{width}")
    os.makedirs(outdir, exist_ok=True)

    seen = set()
    attempts = 0

    # 只要 collected < n，就一直循环
    while len(seen) < n:
        attempts += 1
        try:
            img = fetch_random_image(height, width)
        except Exception as e:
            print(f"⚠️ 第 {attempts} 次尝试下载失败：{e}")
            continue

        hval = image_hash(img)
        if hval in seen:
            print(f"⚠️ 第 {attempts} 次尝试得到重复图片，hash={hval}")
            continue

        # 新图片，保存之
        idx = len(seen) + 1
        seen.add(hval)
        filename = f"{idx:03d}_{hval}.png"
        path = os.path.join(outdir, filename)
        img.save(path)
        print(f"✅ 保存第 {idx}/{n} 张 → {path}")

    print(f"\n共尝试 {attempts} 次，成功去重并保存 {len(seen)} 张图片，位于 “{outdir}/” 目录。")

if __name__ == "__main__":
    # 举例：目标 1000 张 128×128 大小的图片
    generate_and_save(n=50, height=1920, width=1080)
