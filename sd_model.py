"""Stable Diffusion 1.5 推理模块。

负责模型加载、参数处理与图片生成，不涉及任何 HTTP / base64 传输逻辑。
对外输入输出均为 PIL.Image。

支持两种生成模式：
- txt2img：仅文本生成图片（StableDiffusionPipeline）
- img2img：图片+文本生成图片（StableDiffusionImg2ImgPipeline）

支持多个 SD 1.5 架构模型切换，模型从阿里魔搭社区（ModelScope）下载，
国内服务器稳定性远优于 hf-mirror。
"""
import os
import time
import logging
from typing import Optional, Tuple, Dict, Any, Union, List

import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)

# 尝试导入 modelscope（国内阿里魔搭社区，下载稳定）
try:
    from modelscope import snapshot_download as _ms_snapshot_download
    _HAS_MODELSCOPE = True
except ImportError:
    _HAS_MODELSCOPE = False

logger = logging.getLogger("sd-model")

# ---- 可选模型注册表 ----
# 均为标准 SD 1.5 架构（unet/vae/text_encoder），可共用同一套加载逻辑。
# key 为 ModelScope 模型 ID，value 为展示信息。
MODELS: Dict[str, Dict[str, str]] = {
    "AI-ModelScope/realistic-vision-v51": {
        "name": "Realistic Vision v5.1",
        "description": "写实人像与场景，照片级质感",
    },
    "AI-ModelScope/openjourney": {
        "name": "OpenJourney",
        "description": "Midjourney 风格，提示词建议加 mdjrny-v4 style 前缀",
    },
}
DEFAULT_MODEL_ID = "AI-ModelScope/realistic-vision-v51"
# 向后兼容别名（app.py 旧代码可能引用）
MODEL_ID = DEFAULT_MODEL_ID

DEVICE = "cuda"
DTYPE = torch.float16

# 采样器名称 → 构造函数的映射，供前端选择
SCHEDULERS = {
    "dpmpp_2m_karras": ("DPM++ 2M Karras", lambda c: DPMSolverMultistepScheduler.from_config(
        c, use_karras_sigmas=True, algorithm_type="dpmsolver++", solver_order=2)),
    "dpmpp_sde_karras": ("DPM++ SDE Karras", lambda c: DPMSolverMultistepScheduler.from_config(
        c, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", solver_order=2)),
    "euler_a": ("Euler a", lambda c: EulerAncestralDiscreteScheduler.from_config(c)),
    "euler": ("Euler", lambda c: EulerDiscreteScheduler.from_config(c)),
    "dpmpp_2m": ("DPM++ 2M", lambda c: DPMSolverMultistepScheduler.from_config(
        c, algorithm_type="dpmsolver++", solver_order=2)),
    "ddim": ("DDIM", lambda c: DDIMScheduler.from_config(c)),
    "unipc": ("UniPC", lambda c: UniPCMultistepScheduler.from_config(c)),
    "heun": ("Heun", lambda c: HeunDiscreteScheduler.from_config(c)),
    "lms": ("LMS", lambda c: LMSDiscreteScheduler.from_config(c)),
    "pndm": ("PNDM", lambda c: PNDMScheduler.from_config(c)),
}
DEFAULT_SCHEDULER = "dpmpp_2m_karras"

# ModelScope 缓存目录
MODELSCOPE_CACHE = os.environ.get(
    "MODELSCOPE_CACHE",
    os.path.expanduser("~/.cache/modelscope/hub"),
)


class ModelScopeNotInstalledError(RuntimeError):
    """modelscope 包未安装。"""


class CUDAOutOfMemoryError(RuntimeError):
    """GPU 显存不足。app 层据此返回 507，无需导入 torch。"""


class UnknownModelError(KeyError):
    """请求的模型不在 MODELS 注册表中。"""


DEFAULT_STRENGTH = 0.75
DEFAULT_GUIDANCE = 7.5
DEFAULT_STEPS = 50
DEFAULT_NEGATIVE = "lowres, blurry, distorted, watermark, text, low quality, deformed, ugly, bad anatomy, extra limbs, missing fingers, mutated hands"

# 多模型 pipeline 缓存：model_id -> {"img2img": pipe, "txt2img": pipe}
# 两个 pipeline 共享同一套权重（UNet/VAE/text_encoder），所以同时加载不额外占显存。
_pipelines: Dict[str, Dict[str, Any]] = {}
_active_model_id: Optional[str] = None


def cuda_available() -> bool:
    return torch.cuda.is_available()


def is_loaded() -> bool:
    return _active_model_id is not None and _active_model_id in _pipelines


def has_modelscope() -> bool:
    return _HAS_MODELSCOPE


def get_models() -> Dict[str, Any]:
    """返回所有可选模型及当前激活模型，供前端 / /models 端点使用。"""
    return {
        "default": DEFAULT_MODEL_ID,
        "active": _active_model_id,
        "models": {
            mid: {"name": info["name"], "description": info["description"]}
            for mid, info in MODELS.items()
        },
    }


def get_active_model_id() -> Optional[str]:
    return _active_model_id


def _resolve_model_id(model_id: Optional[str]) -> str:
    """校验并解析模型 ID，None 或空则返回当前激活模型或默认模型。"""
    if not model_id:
        return _active_model_id or DEFAULT_MODEL_ID
    if model_id not in MODELS:
        raise UnknownModelError(
            "未知模型: {}，可用模型: {}".format(model_id, ", ".join(MODELS.keys()))
        )
    return model_id


def get_img2img_pipeline(model_id: Optional[str] = None) -> StableDiffusionImg2ImgPipeline:
    """获取指定模型的 img2img pipeline（若未加载则触发加载并切换为激活模型）。"""
    mid = _resolve_model_id(model_id)
    _ensure_model_loaded(mid)
    return _pipelines[mid]["img2img"]


def get_txt2img_pipeline(model_id: Optional[str] = None) -> StableDiffusionPipeline:
    """获取指定模型的 txt2img pipeline（复用 img2img 的组件，零额外显存）。"""
    mid = _resolve_model_id(model_id)
    _ensure_model_loaded(mid)
    entry = _pipelines[mid]
    if entry.get("txt2img") is None:
        img2img = entry["img2img"]
        entry["txt2img"] = StableDiffusionPipeline(
            vae=img2img.vae,
            text_encoder=img2img.text_encoder,
            tokenizer=img2img.tokenizer,
            unet=img2img.unet,
            scheduler=img2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(DEVICE)
        entry["txt2img"].safety_checker = None
    return entry["txt2img"]


def _model_local_dir(model_id: str) -> str:
    """返回模型在 ModelScope 缓存中的本地目录路径。"""
    return os.path.join(MODELSCOPE_CACHE, model_id)


def _log_dir_contents(path: str, prefix: str = "    ") -> None:
    """递归打印目录内容，用于诊断。"""
    for root, _dirs, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, path).replace("\\", "/")
            try:
                size_mb = os.path.getsize(full) / 1024 / 1024
                logger.info("%s%-60s %8.1f MB", prefix, rel, size_mb)
            except OSError:
                logger.info("%s%s (size unknown)", prefix, rel)


def _check_model_files(model_dir: str) -> Dict[str, bool]:
    """检查模型目录中各组件的权重文件存在情况。"""
    components = {
        "text_encoder": ["model.fp16.safetensors", "model.safetensors",
                         "pytorch_model.fp16.bin", "pytorch_model.bin"],
        "unet": ["diffusion_pytorch_model.fp16.safetensors",
                 "diffusion_pytorch_model.safetensors",
                 "diffusion_pytorch_model.fp16.bin",
                 "diffusion_pytorch_model.bin"],
        "vae": ["diffusion_pytorch_model.fp16.safetensors",
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.fp16.bin",
                "diffusion_pytorch_model.bin"],
    }
    checks = {}
    for comp, files in components.items():
        for f in files:
            key = "{}/{}".format(comp, f)
            checks[key] = os.path.isfile(os.path.join(model_dir, comp, f))
    return checks


def _has_complete_weights(model_dir: str) -> bool:
    """检查模型目录是否包含至少一套完整的权重文件（text_encoder + unet + vae）。"""
    checks = _check_model_files(model_dir)
    has_text = any(v for k, v in checks.items() if k.startswith("text_encoder/"))
    has_unet = any(v for k, v in checks.items() if k.startswith("unet/"))
    has_vae = any(v for k, v in checks.items() if k.startswith("vae/"))
    return has_text and has_unet and has_vae


def _download_with_retry(model_id: str) -> str:
    """通过 ModelScope 下载指定模型，带重试与缓存校验。"""
    if not _HAS_MODELSCOPE:
        raise ModelScopeNotInstalledError(
            "modelscope 包未安装。请先运行:\n"
            "  pip install modelscope"
        )

    model_dir = _model_local_dir(model_id)

    # 检查现有缓存是否完整
    if os.path.isdir(model_dir) and _has_complete_weights(model_dir):
        logger.info("[1/3] Cache hit, model files present at: %s", model_dir)
        return model_dir

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        logger.info("[1/3] Downloading %s from ModelScope (attempt %d/%d, 进度条见下方)...",
                    model_id, attempt, max_attempts)
        try:
            local_path = _ms_snapshot_download(model_id)
            if not _has_complete_weights(local_path):
                raise OSError(
                    "下载完成但未找到完整权重文件（text_encoder + unet + vae）"
                )
            return local_path
        except Exception as e:
            logger.warning("[1/3] Attempt %d failed: %s: %s",
                           attempt, type(e).__name__, str(e)[:200])
            if attempt >= max_attempts:
                logger.error("[1/3] All %d attempts exhausted.", max_attempts)
                raise
            backoff = 5 * attempt
            logger.info("[1/3] Retrying in %ds (已下载文件会跳过)...", backoff)
            time.sleep(backoff)
    raise RuntimeError("download failed unexpectedly")


def _load_pipeline(model_dir: str) -> StableDiffusionImg2ImgPipeline:
    """从本地目录加载 pipeline，自动探测并选择最佳文件格式。

    按优先级依次尝试: fp16 safetensors > fp32 safetensors > fp16 bin > fp32 bin
    """
    checks = _check_model_files(model_dir)
    logger.info("[2/3] Available weight files:")
    for k in sorted(checks.keys()):
        if checks[k]:
            logger.info("    ✓ %s", k)

    def _all_exist(comp, prefix_list):
        return all(
            checks.get("{}/{}".format(comp, f), False)
            for f in prefix_list
        )

    strategies = [
        ("fp16 safetensors",
         {"variant": "fp16", "use_safetensors": True},
         [("text_encoder", "model.fp16.safetensors"),
          ("unet", "diffusion_pytorch_model.fp16.safetensors"),
          ("vae", "diffusion_pytorch_model.fp16.safetensors")]),
        ("fp32 safetensors",
         {"use_safetensors": True},
         [("text_encoder", "model.safetensors"),
          ("unet", "diffusion_pytorch_model.safetensors"),
          ("vae", "diffusion_pytorch_model.safetensors")]),
        ("fp16 bin",
         {"variant": "fp16"},
         [("text_encoder", "pytorch_model.fp16.bin"),
          ("unet", "diffusion_pytorch_model.fp16.bin"),
          ("vae", "diffusion_pytorch_model.fp16.bin")]),
        ("fp32 bin (default)",
         {},
         [("text_encoder", "pytorch_model.bin"),
          ("unet", "diffusion_pytorch_model.bin"),
          ("vae", "diffusion_pytorch_model.bin")]),
    ]

    last_error = None
    for name, kwargs, required in strategies:
        if not all(checks.get("{}/{}".format(c, f), False) for c, f in required):
            logger.info("[2/3] Skipping '%s' (files not present)", name)
            continue
        logger.info("[2/3] Trying load strategy: %s ...", name)
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_dir,
                torch_dtype=DTYPE,
                safety_checker=None,
                requires_safety_checker=False,
                feature_extractor=None,
                local_files_only=True,
                **kwargs,
            )
            pipe.safety_checker = None
            logger.info("[2/3] ✓ Loaded with strategy: %s", name)
            return pipe
        except Exception as e:
            logger.warning("[2/3] Strategy '%s' failed: %s", name, str(e)[:150])
            last_error = e

    raise RuntimeError("所有加载策略均失败。最后错误: {}".format(last_error))


def _set_scheduler(pipe, scheduler_name: str) -> None:
    """切换 pipeline 的采样器。在每次生成前调用。"""
    if scheduler_name not in SCHEDULERS:
        scheduler_name = DEFAULT_SCHEDULER
    _, factory = SCHEDULERS[scheduler_name]
    pipe.scheduler = factory(pipe.scheduler.config)


def _ensure_model_loaded(model_id: str) -> None:
    """确保指定模型已加载到缓存并设为激活模型。已加载则直接返回。"""
    global _active_model_id
    if model_id in _pipelines:
        _active_model_id = model_id
        return
    # 加载新模型（init 内部处理下载/加载/迁 GPU/质量优化）
    init(model_id)


def init(model_id: Optional[str] = None) -> StableDiffusionImg2ImgPipeline:
    """主动下载并加载指定模型并设为激活模型。服务启动时调用，避免首次请求等待。

    若 model_id 为 None，则加载默认模型；若已有激活模型则保持。
    """
    global _active_model_id
    mid = _resolve_model_id(model_id)

    # 已加载则直接切为激活
    if mid in _pipelines:
        _active_model_id = mid
        logger.info("Model already loaded, switch active to: %s", mid)
        return _pipelines[mid]["img2img"]

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，当前为强制 CUDA 模式，无法运行。")

    info = MODELS[mid]
    logger.info("=" * 60)
    logger.info("Initializing Stable Diffusion pipeline")
    logger.info("  model_id        : %s", mid)
    logger.info("  model_name      : %s", info["name"])
    logger.info("  device          : %s", DEVICE)
    logger.info("  dtype           : %s", DTYPE)
    logger.info("  modelscope_cache: %s", MODELSCOPE_CACHE)
    logger.info("  has_modelscope  : %s", _HAS_MODELSCOPE)
    logger.info("=" * 60)

    # ---- Stage 1: 下载权重 ----
    t0 = time.time()
    local_path = _download_with_retry(mid)
    logger.info("[1/3] Download complete in %.1fs", time.time() - t0)
    logger.info("[1/3] Cached at: %s", local_path)
    _log_dir_contents(local_path)

    # ---- Stage 2: 从本地加载 pipeline（自动探测文件格式）----
    t1 = time.time()
    logger.info("[2/3] Loading pipeline from local cache...")
    pipe = _load_pipeline(local_path)
    logger.info("[2/3] Pipeline loaded in %.1fs", time.time() - t1)

    # ---- Stage 3: 迁移到 GPU ----
    t2 = time.time()
    logger.info("[3/3] Moving pipeline to %s...", DEVICE)
    pipe = pipe.to(DEVICE)
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    logger.info("[3/3] Pipeline on %s in %.1fs", DEVICE, time.time() - t2)

    # ---- Stage 4: 质量优化 ----
    # 4a. VAE 精度 —— fp16 下 VAE 编/解码会出现色彩偏差/黑块。
    #     diffusers 的 upcast_vae() 会把 VAE 权重升到 fp32 并在 forward 时自动处理 dtype。
    try:
        pipe.upcast_vae()
        logger.info("[4/4] VAE upcasted to fp32 via upcast_vae() (fixes fp16 color artifacts)")
    except Exception as e:
        logger.warning("[4/4] VAE upcast failed: %s", e)

    # 4b. 切换默认采样器到 DPM++ 2M Karras（远优于 SD1.5 默认的 PNDM）
    try:
        _set_scheduler(pipe, DEFAULT_SCHEDULER)
        logger.info("[4/4] Scheduler set to: %s (%s)",
                    DEFAULT_SCHEDULER, SCHEDULERS[DEFAULT_SCHEDULER][0])
    except Exception as e:
        logger.warning("[4/4] Scheduler switch failed: %s", e)

    # GPU 信息
    try:
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB total)", props.name, props.total_memory / 1024**3)
    except Exception:
        pass

    # 写入缓存并设为激活
    _pipelines[mid] = {"img2img": pipe, "txt2img": None}
    _active_model_id = mid

    total = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        "Pipeline READY. model=%s total=%.1fs safety_checker=%s (disabled, no content filter).",
        mid, total, pipe.safety_checker,
    )
    logger.info("=" * 60)
    return pipe


def _resize_init_image(image: Image.Image, max_side: int) -> Tuple[Image.Image, Tuple[int, int]]:
    """等比缩放原图，使最长边不超过 max_side，且每边不小于 64。"""
    if max_side < 64:
        max_side = 64
    ratio = min(max_side / image.width, max_side / image.height)
    new_size = (max(64, int(image.width * ratio)),
                max(64, int(image.height * ratio)))
    return image.resize(new_size, Image.LANCZOS), new_size


def _build_generator(seed: Optional[Any]) -> Tuple[torch.Generator, int]:
    if seed is None or seed == "":
        gen = torch.Generator(device=DEVICE)
        return gen, gen.seed()
    seed_used = int(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed_used)
    return gen, seed_used


def generate(
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    strength: float = DEFAULT_STRENGTH,
    guidance_scale: float = DEFAULT_GUIDANCE,
    num_inference_steps: int = DEFAULT_STEPS,
    seed: Optional[Any] = None,
    max_side: int = 512,
    scheduler: str = DEFAULT_SCHEDULER,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """对输入图片执行 img2img 生成。"""
    t_start = time.time()
    mid = _resolve_model_id(model)

    init_image = init_image.convert("RGB")
    logger.info("Input image: %dx%d", init_image.width, init_image.height)
    init_image, new_size = _resize_init_image(init_image, max_side)
    logger.info("Resized to: %dx%d", new_size[0], new_size[1])

    gen, seed_used = _build_generator(seed)
    pipe = get_img2img_pipeline(mid)
    _set_scheduler(pipe, scheduler)

    logger.info(
        "[img2img] model=%s size=%s steps=%d strength=%.2f cfg=%.1f seed=%d scheduler=%s prompt=%r",
        mid, new_size, num_inference_steps, strength, guidance_scale, seed_used, scheduler, prompt[:80],
    )

    t_infer = time.time()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=gen,
        )
    except torch.cuda.OutOfMemoryError:
        release_memory()
        raise CUDAOutOfMemoryError("GPU 显存不足")
    logger.info("Inference done in %.2fs", time.time() - t_infer)

    out_image = result.images[0]
    logger.info("Total generate time: %.2fs", time.time() - t_start)

    return {
        "image": out_image,
        "seed": seed_used,
        "params": {
            "mode": "img2img",
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "scheduler": scheduler,
            "size": list(new_size),
            "model": mid,
            "device": DEVICE,
        },
    }


def generate_txt2img(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    guidance_scale: float = DEFAULT_GUIDANCE,
    num_inference_steps: int = DEFAULT_STEPS,
    seed: Optional[Any] = None,
    width: int = 512,
    height: int = 512,
    scheduler: str = DEFAULT_SCHEDULER,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """仅文本生成图片（txt2img）。"""
    t_start = time.time()
    mid = _resolve_model_id(model)

    # SD 要求尺寸为 8 的倍数
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)

    gen, seed_used = _build_generator(seed)
    pipe = get_txt2img_pipeline(mid)
    _set_scheduler(pipe, scheduler)

    logger.info(
        "[txt2img] model=%s %dx%d steps=%d cfg=%.1f seed=%d scheduler=%s prompt=%r",
        mid, width, height, num_inference_steps, guidance_scale, seed_used, scheduler, prompt[:80],
    )

    t_infer = time.time()
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=gen,
        )
    except torch.cuda.OutOfMemoryError:
        release_memory()
        raise CUDAOutOfMemoryError("GPU 显存不足")
    logger.info("Inference done in %.2fs", time.time() - t_infer)

    out_image = result.images[0]
    logger.info("Total generate time: %.2fs", time.time() - t_start)

    return {
        "image": out_image,
        "seed": seed_used,
        "params": {
            "mode": "txt2img",
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "scheduler": scheduler,
            "size": [width, height],
            "model": mid,
            "device": DEVICE,
        },
    }


def release_memory() -> None:
    """OOM 后清理显存缓存。"""
    torch.cuda.empty_cache()
