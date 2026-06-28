# Stable Diffusion 1.5 图生图 Web 服务

基于 `AI-ModelScope/stable-diffusion-v1-5`（fp16）的 Stable Diffusion Web 服务，支持两种生成模式：

- **仅文本生成（txt2img）**：输入提示词，从随机噪声生成图片
- **图片+文本生成（img2img）**：上传参考图 + 提示词，在原图基础上重绘

## 环境

- Python 3.10（已使用项目内 `venv/`，含 torch 2.9.1+cu128、diffusers 0.38.0、transformers 5.12.1）
- NVIDIA GPU + CUDA 12.8（强制 CUDA 模式，支持 RTX 5070 Ti 的 sm_120 架构）
- Flask 3.x
- modelscope（阿里魔搭社区 SDK，用于模型下载）

> **GPU 兼容性**：RTX 50 系列（Blackwell，sm_120）需要 torch cu128+ 版本。
> PyPI 默认的 torch 是 CPU 版或旧 CUDA 版，不支持 sm_120。
> 如需重装 torch，参考下方"重装 torch"部分。

> **网络要求**：模型从阿里魔搭社区（modelscope.cn）下载。如果你的网络需要代理，
> 启动前请确保代理已开启并正确配置环境变量：
> ```bash
> # Windows CMD
> set https_proxy=http://127.0.0.1:你的代理端口
> set http_proxy=http://127.0.0.1:你的代理端口
> ```

## 文件结构

```
stable-diffusion-1.5-test/
├── app.py                  # Flask 主程序（HTTP 路由 + base64 传输）
├── sd_model.py             # 模型推理模块（ModelScope 下载 + diffusers 加载 + img2img 生成）
├── templates/
│   └── index.html          # 前端页面（上传图 + 文本输入）
├── requirements.txt
└── venv/                   # 已有虚拟环境
```

## 安装依赖

```bash
# 1. 安装 modelscope（如网络需要代理，先设置代理环境变量）
venv\Scripts\python.exe -m pip install modelscope

# 2. 安装 Flask（如尚未安装）
venv\Scripts\python.exe -m pip install flask
```

## 重装 torch（仅 RTX 50 系列需要）

RTX 5070 Ti（Blackwell，sm_120）需要 torch cu128 版本。如需重装：

```bash
# 1. 下载 wheel（用项目内脚本，支持断点续传）
venv\Scripts\python.exe download_torch.py

# 2. 卸载旧版 torchvision / torchaudio（不兼容新 torch）
venv\Scripts\python.exe -m pip uninstall torchvision torchaudio -y

# 3. 安装新 torch
venv\Scripts\python.exe -m pip install --no-deps "torch-2.9.1+cu128-cp310-cp310-win_amd64.whl"
```

> torchvision / torchaudio 卸载后不影响 SD 推理，transformers 会自动 fallback 到 PIL 后端。

## 运行

```bash
# Windows（推荐，直接用 venv）
venv\Scripts\python.exe app.py

# WSL 调用 Windows venv（需确保代理对 Windows Python 可用）
venv/Scripts/python.exe app.py
```

首次启动后，浏览器访问 <http://localhost:5000>。

### 配置项（环境变量）

| 变量                | 默认值                                    | 说明                              |
| ------------------- | ----------------------------------------- | --------------------------------- |
| `SD_MODEL_ID`       | `AI-ModelScope/stable-diffusion-v1-5`     | ModelScope 模型仓库 ID            |
| `MODELSCOPE_CACHE`  | `~/.cache/modelscope/hub`                 | 模型缓存目录                      |
| `HOST`              | `0.0.0.0`                                 | 监听地址                          |
| `PORT`              | `5000`                                    | 监听端口                          |
| `https_proxy`       | -                                         | 代理地址（如需代理访问网络）      |

首次运行时从 ModelScope 下载模型权重（约 4GB），内置 5 次重试，下载完成后自动加载到 GPU。

## 架构

```
app.py (HTTP 层)              sd_model.py (模型层)
┌─────────────────┐          ┌──────────────────────────┐
│ Flask 路由      │          │ init()                   │
│  GET  /         │  ──────► │   ├─ ModelScope 下载     │
│  POST /generate │          │   ├─ 探测文件格式         │
│  GET  /health   │          │   ├─ diffusers 加载       │
│                 │          │   └─ 迁移到 GPU           │
│ base64 ↔ PIL    │          │                          │
│ 请求解析/校验   │          │ generate()               │
│ 响应格式化      │          │   ├─ 图片缩放            │
│                 │          │   ├─ img2img 推理         │
│                 │  ◄────── │   └─ 返回 PIL.Image       │
└─────────────────┘          └──────────────────────────┘
```

模型加载器自动探测 ModelScope 仓库中的文件格式，按优先级尝试：
1. fp16 safetensors（最快、最省显存）
2. fp32 safetensors
3. fp16 bin
4. fp32 bin

## API

### `GET /`
前端页面（含模式切换：仅文本 / 图片+文本）。

### `POST /generate`
根据是否传 `image` 字段自动选择模式：

**txt2img 模式**（不传 image）：
```json
{
  "prompt": "提示词（必填）",
  "negative_prompt": "负向提示词",
  "width": 512,
  "height": 512,
  "guidance_scale": 7.5,
  "num_inference_steps": 30,
  "seed": ""
}
```

**img2img 模式**（传 image）：
```json
{
  "prompt": "提示词（必填）",
  "negative_prompt": "负向提示词",
  "image": "data:image/png;base64,....（base64 data URL）",
  "strength": 0.75,
  "guidance_scale": 7.5,
  "num_inference_steps": 30,
  "max_side": 512,
  "seed": ""
}
```

响应：

```json
{
  "image": "data:image/png;base64,...",
  "seed": 12345,
  "params": { "mode": "txt2img", "guidance_scale": 7.5, "num_inference_steps": 30, "size": [512, 512], "model": "...", "device": "cuda" }
}
```

### `GET /health`
健康检查，返回模型、CUDA、modelscope 安装状态。

## 参数说明

| 参数                  | 范围       | 作用                                       |
| --------------------- | ---------- | ------------------------------------------ |
| `strength`            | 0~1        | 重绘强度。越大越偏离原图，1=完全重绘       |
| `guidance_scale`      | 1~20       | CFG 引导系数。越大越贴合提示词             |
| `num_inference_steps` | 1~100      | 推理步数。越多质量越好，越慢               |
| `max_side`            | 64~1024    | 原图等比缩放后的最长边像素                 |
| `seed`                | 整数/空    | 相同种子+参数可复现结果                    |
