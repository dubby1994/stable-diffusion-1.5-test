"""Stable Diffusion 1.5 图生图 Web 服务入口。

仅负责 HTTP 路由、请求解析与 base64 图片传输，
模型推理逻辑见 sd_model.py。
"""
import io
import os

# 修复 WSL 调用 Windows Python 时的代理问题：
# WSL 的 https_proxy=localhost:3128 对 Windows 不可达，导致所有 HTTPS 失败。
# 但直接清空也不行——modelscope 用的 requests 库不读 Windows 注册表代理，会直连失败。
# 正确做法：清掉无效的 WSL 代理，从 Windows 注册表读出真实代理并设回环境变量。
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "ALL_PROXY", "ftp_proxy", "FTP_PROXY"):
    os.environ.pop(_k, None)

try:
    # urllib.request.getproxies() 会读 Windows 注册表的代理设置
    import urllib.request as _ur
    _win_proxies = _ur.getproxies()
    # 代理服务器本身说的是 HTTP 协议（即使代理 HTTPS 流量也用 CONNECT 隧道），
    # 所以 http_proxy 和 https_proxy 都必须用 http:// 前缀，不能用 https://。
    _proxy = _win_proxies.get("https") or _win_proxies.get("http")
    if _proxy:
        _proxy = _proxy.replace("https://", "http://")
        os.environ["http_proxy"] = _proxy
        os.environ["https_proxy"] = _proxy
except Exception:
    pass

import time
import base64
import logging
from datetime import datetime

from PIL import Image
from flask import Flask, request, jsonify, render_template

import sd_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sd-service")

# 静默 httpx 的逐请求日志，避免淹没下载进度条与关键日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static", template_folder="templates")


def _decode_image(data_url: str) -> Image.Image:
    """将 base64 data URL（或裸 base64）解码为 PIL.Image。"""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image(img: Image.Image, fmt: str = "PNG") -> str:
    """将 PIL.Image 编码为 base64 data URL。"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return "data:image/{};base64,{}".format(
        fmt.lower(), base64.b64encode(buf.getvalue()).decode("ascii")
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    t_req = time.time()
    try:
        payload = request.get_json(force=True, silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        image_data = (payload.get("image") or "").strip()

        if not prompt:
            return jsonify({"error": "prompt 不能为空"}), 400

        if image_data:
            # ---- img2img 模式 ----
            logger.info("Request /generate [img2img] | model=%s prompt=%r image_bytes=%d",
                        payload.get("model"), prompt[:60], len(image_data))
            init_image = _decode_image(image_data)

            result = sd_model.generate(
                init_image=init_image,
                prompt=prompt,
                negative_prompt=(payload.get("negative_prompt") or sd_model.DEFAULT_NEGATIVE).strip(),
                strength=float(payload.get("strength") or sd_model.DEFAULT_STRENGTH),
                guidance_scale=float(payload.get("guidance_scale") or sd_model.DEFAULT_GUIDANCE),
                num_inference_steps=int(payload.get("num_inference_steps") or sd_model.DEFAULT_STEPS),
                seed=payload.get("seed"),
                max_side=int(payload.get("max_side") or 512),
                scheduler=payload.get("scheduler") or sd_model.DEFAULT_SCHEDULER,
                model=payload.get("model"),
            )
        else:
            # ---- txt2img 模式 ----
            logger.info("Request /generate [txt2img] | model=%s prompt=%r",
                        payload.get("model"), prompt[:60])

            # 尺寸：优先用 width/height，否则用 max_side 作为正方形边长
            width = int(payload.get("width") or 0)
            height = int(payload.get("height") or 0)
            if width <= 0 or height <= 0:
                side = int(payload.get("max_side") or 512)
                width = height = side

            result = sd_model.generate_txt2img(
                prompt=prompt,
                negative_prompt=(payload.get("negative_prompt") or sd_model.DEFAULT_NEGATIVE).strip(),
                guidance_scale=float(payload.get("guidance_scale") or sd_model.DEFAULT_GUIDANCE),
                num_inference_steps=int(payload.get("num_inference_steps") or sd_model.DEFAULT_STEPS),
                seed=payload.get("seed"),
                width=width,
                height=height,
                scheduler=payload.get("scheduler") or sd_model.DEFAULT_SCHEDULER,
                model=payload.get("model"),
            )

        return jsonify({
            "image": _encode_image(result["image"], "PNG"),
            "seed": result["seed"],
            "params": result["params"],
        })

    except sd_model.CUDAOutOfMemoryError:
        logger.exception("CUDA OOM")
        return jsonify({"error": "GPU 显存不足，请降低图片尺寸或减小 steps"}), 507
    except sd_model.UnknownModelError as e:
        logger.warning("Unknown model: %s", e)
        return jsonify({"error": str(e)}), 400
    except sd_model.ModelScopeNotInstalledError as e:
        logger.error("modelscope not installed: %s", e)
        return jsonify({"error": "modelscope 包未安装，请先运行: pip install modelscope"}), 503
    except Exception as e:
        logger.exception("generate failed")
        return jsonify({"error": "生成失败: {}".format(e)}), 500
    finally:
        logger.info("Request /generate finished in %.2fs",
                    time.time() - t_req)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": sd_model.get_active_model_id(),
        "device": sd_model.DEVICE,
        "pipeline_loaded": sd_model.is_loaded(),
        "cuda_available": sd_model.cuda_available(),
        "modelscope_installed": sd_model.has_modelscope(),
        "time": datetime.now().isoformat(),
    })


@app.route("/models")
def models():
    return jsonify(sd_model.get_models())


@app.route("/schedulers")
def schedulers():
    return jsonify({
        "default": sd_model.DEFAULT_SCHEDULER,
        "schedulers": {k: v[0] for k, v in sd_model.SCHEDULERS.items()},
    })


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))

    logger.info("=" * 60)
    logger.info("Starting Stable Diffusion img2img service")
    logger.info("  listen : %s:%d", host, port)
    logger.info("=" * 60)

    # 服务启动后立刻下载并加载模型（用户要求）
    logger.info("Pre-loading model (will download if not cached)...")
    try:
        sd_model.init()
    except Exception:
        logger.exception("Model init failed! Server will still start, "
                         "model will retry lazy-load on first /generate request.")

    logger.info("HTTP server starting on %s:%d", host, port)
    app.run(host=host, port=port, threaded=False)
