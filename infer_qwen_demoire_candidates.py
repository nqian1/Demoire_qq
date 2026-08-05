#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit-2509 去摩尔纹推理脚本

功能：
1. 支持单张图片推理；
2. 支持读取 Flow-Factory 风格的 JSONL：
       {"prompt": "...", "image": "000000.jpg"}
3. 支持纯基模、基模+LoRA，以及两者公平对比；
4. Base 与 LoRA 使用相同输入、prompt、seed、分辨率、steps 和 CFG；
5. 自动按输入图片宽高比计算输出尺寸，目标面积默认为 1024x1024；
6. 保存推理结果、对比图和 metadata.jsonl。

推荐运行环境：
    在 Flow-Factory 的 Python 环境中运行。

示例：
    # 单张图，只测基模
    python tools/infer_qwen_demoire.py \
      --mode base \
      --input_image dataset/moire/000000.jpg \
      --prompt "识别并去除投影屏幕显示区域中的摩尔纹，严格保持文字、Logo和背景细节不变。" \
      --output_dir outputs/base_test

    # 测试 test.jsonl，只测基模
    python tools/infer_qwen_demoire.py \
      --mode base \
      --jsonl dataset/moire/test.jsonl \
      --output_dir outputs/base_test

    # 同时对比基模和 LoRA
    python tools/infer_qwen_demoire.py \
      --mode both \
      --jsonl dataset/moire/test.jsonl \
      --lora_path saves/你的运行目录/checkpoint-xxx \
      --output_dir outputs/base_vs_lora
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError as exc:
    raise ImportError(
        "无法导入 QwenImageEditPlusPipeline。请确认当前环境安装了项目要求的 diffusers，"
        "并在 Flow-Factory 的环境中运行。"
    ) from exc


DEFAULT_MODEL_PATH = "/data/ckpts/Qwen/Qwen-Image-Edit-2509/"

DEFAULT_PROMPT = (
    "识别并去除图像显示区域中所有明显的斜向波纹、网格状摩尔纹及周期性彩色干涉条纹，"
    "要求高保真修复，完整保留文字内容、Logo、边缘锐度、颜色、背景图案和细节；"
    "避免画面模糊、涂抹、塑料感、锐化光晕或虚假纹理。"
    "仅修复摩尔纹及其相关伪影，除摩尔纹区域外，其他区域禁止进行任何改动。"
)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    prompt: str
    sample_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit-2509 去摩尔纹 Base/LoRA 推理与公平对比脚本"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input_image",
        type=Path,
        help="单张输入图片路径。",
    )
    source.add_argument(
        "--jsonl",
        type=Path,
        help='JSONL 路径，每行格式：{"prompt": "...", "image": "000000.jpg"}。',
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="单张图片使用的 prompt；未提供时使用脚本内置通用去摩尔纹 prompt。",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=None,
        help="JSONL 中相对图片路径的根目录。默认使用 JSONL 所在目录。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理多少个样本；默认处理全部。",
    )

    parser.add_argument(
        "--mode",
        choices=("base", "lora", "both"),
        default="base",
        help="base=纯基模，lora=基模+LoRA，both=同时生成并保存对比图。",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(DEFAULT_MODEL_PATH),
        help=f"基模目录，默认：{DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--lora_path",
        type=Path,
        default=None,
        help="Flow-Factory 保存的 LoRA checkpoint 目录。mode=lora/both 时必须提供。",
    )
    parser.add_argument(
        "--lora_weight_name",
        type=str,
        default=None,
        help="LoRA 权重文件名。通常不需要填写；目录中有多个权重文件时再指定。",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="demoire",
        help="加载到 Diffusers Pipeline 中的 LoRA adapter 名称。",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA 强度，默认 1.0。",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="结果保存目录。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="目标输出面积的边长，默认 1024，即目标面积约为 1024x1024。",
    )
    parser.add_argument(
        "--auto_resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按输入图宽高比自动计算输出尺寸，默认开启。",
    )
    parser.add_argument(
        "--preserve_resolution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Preserve the exact input resolution. Inputs are edge-padded to a multiple "
            "of 32 for Qwen inference and cropped back after generation."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="关闭 auto_resize 后使用的固定高度。",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="关闭 auto_resize 后使用的固定宽度。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="推理步数，默认与配置 eval.num_inference_steps=20 一致。",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Qwen-Image-Edit-Plus 的真实 CFG，默认与配置 guidance_scale=4 一致。",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        help="负向提示。默认单个空格，用于启用 true CFG。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认与配置 eval.seed=42 一致。",
    )
    parser.add_argument(
        "--increment_seed",
        action="store_true",
        help="第 i 张图片使用 seed+i；未开启时所有图片都使用同一个 seed。",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=4,
        help="Number of stochastic candidates generated for each input image (default: 4).",
    )
    parser.add_argument(
        "--bank_version",
        type=str,
        default="qwen-base-v1",
        help="Version string written to every pseudo-GT candidate metadata record.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=1024,
        help="文本最大长度，默认 1024。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="普通单卡加载时使用的设备，例如 cuda:0。",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help='多卡自动切分，例如 "balanced" 或 "auto"。提供后不再调用 pipe.to(device)。',
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="启用 model CPU offload，显存不足时使用；不能和 --device_map 同时使用。",
    )
    parser.add_argument(
        "--vae_tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="开启 VAE tiling 以降低峰值显存，默认开启。",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="模型精度，默认 bf16。",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="只读取本地模型文件，默认开启。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的输出图片；默认跳过已有结果。",
    )
    parser.add_argument(
        "--save_comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mode=both 时保存 Input/Base/LoRA 横向对比图，默认开启。",
    )
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def calculate_dimensions(target_area: int, height_width_ratio: float) -> tuple[int, int]:
    """
    与 Flow-Factory qwen_image_edit_plus.py 中的 calculate_dimensions 保持一致。
    返回 (width, height)，并对齐到 32 的倍数。
    """
    height = math.sqrt(target_area * height_width_ratio)
    width = height / height_width_ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return max(width, 32), max(height, 32)


def resolve_output_size(
    image: Image.Image,
    auto_resize: bool,
    resolution: int,
    height: int,
    width: int,
) -> tuple[int, int]:
    if auto_resize:
        image_width, image_height = image.size
        out_width, out_height = calculate_dimensions(
            target_area=resolution * resolution,
            height_width_ratio=image_height / image_width,
        )
    else:
        out_width, out_height = width, height

    # Qwen VAE/latent packing 要求尺寸可被较大的块大小整除；32 与项目实现一致。
    out_width = max((out_width // 32) * 32, 32)
    out_height = max((out_height // 32) * 32, 32)
    return out_width, out_height


def prepare_generation_image(
    image: Image.Image,
    preserve_resolution: bool,
    auto_resize: bool,
    resolution: int,
    height: int,
    width: int,
) -> tuple[Image.Image, int, int, Optional[tuple[int, int, int, int]]]:
    """Prepare an aligned Qwen input and an optional crop back to source size."""
    image = image.convert("RGB")
    if not preserve_resolution:
        out_width, out_height = resolve_output_size(
            image=image,
            auto_resize=auto_resize,
            resolution=resolution,
            height=height,
            width=width,
        )
        return image, out_width, out_height, None

    source_width, source_height = image.size
    out_width = max(math.ceil(source_width / 32) * 32, 32)
    out_height = max(math.ceil(source_height / 32) * 32, 32)
    pad_width = out_width - source_width
    pad_height = out_height - source_height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    if pad_width or pad_height:
        array = np.asarray(image)
        array = np.pad(
            array,
            ((top, bottom), (left, right), (0, 0)),
            mode="edge",
        )
        generation_image = Image.fromarray(array)
    else:
        generation_image = image

    crop_box = (left, top, left + source_width, top + source_height)
    return generation_image, out_width, out_height, crop_box


def iter_jsonl_samples(
    jsonl_path: Path,
    image_root: Optional[Path],
    max_samples: Optional[int],
) -> Iterable[Sample]:
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"找不到 JSONL：{jsonl_path}")

    root = image_root if image_root is not None else jsonl_path.parent
    yielded = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_number} 不是合法 JSON：{exc}"
                ) from exc

            image_value = record.get("image")
            if not image_value:
                raise ValueError(
                    f"{jsonl_path}:{line_number} 缺少非空字段 'image'"
                )

            prompt = str(record.get("prompt") or DEFAULT_PROMPT)
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = root / image_path

            sample_id = Path(image_value).stem
            yield Sample(
                image_path=image_path,
                prompt=prompt,
                sample_id=sample_id,
            )

            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                break


def build_samples(args: argparse.Namespace) -> list[Sample]:
    if args.input_image is not None:
        return [
            Sample(
                image_path=args.input_image,
                prompt=args.prompt or DEFAULT_PROMPT,
                sample_id=args.input_image.stem,
            )
        ]

    return list(
        iter_jsonl_samples(
            jsonl_path=args.jsonl,
            image_root=args.image_root,
            max_samples=args.max_samples,
        )
    )


def load_pipeline(args: argparse.Namespace) -> QwenImageEditPlusPipeline:
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"基模目录不存在：{model_path}")

    if args.cpu_offload and args.device_map is not None:
        raise ValueError("--cpu_offload 与 --device_map 不能同时使用。")

    load_kwargs = {
        "torch_dtype": get_torch_dtype(args.dtype),
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map

    print(f"[Load] base model: {model_path}", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(model_path),
        **load_kwargs,
    )

    if args.device_map is None:
        if args.cpu_offload:
            if not torch.cuda.is_available():
                raise RuntimeError("--cpu_offload 需要 CUDA。")
            gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipe.to(args.device)

    if args.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    return pipe


def load_lora(pipe: QwenImageEditPlusPipeline, args: argparse.Namespace) -> None:
    if args.lora_path is None:
        raise ValueError("mode=lora/both 时必须提供 --lora_path。")

    lora_path = args.lora_path.expanduser().resolve()
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA 路径不存在：{lora_path}")

    kwargs = {"adapter_name": args.adapter_name}
    if args.lora_weight_name:
        kwargs["weight_name"] = args.lora_weight_name

    print(f"[Load] LoRA: {lora_path}", flush=True)
    try:
        pipe.load_lora_weights(str(lora_path), **kwargs)
    except Exception as exc:
        files = []
        if lora_path.is_dir():
            files = sorted(p.name for p in lora_path.iterdir())
        raise RuntimeError(
            "LoRA 加载失败。请确认该 checkpoint 是 Diffusers/PEFT 可加载的 LoRA 目录。"
            f"\nLoRA 路径：{lora_path}"
            f"\n目录文件：{files[:30]}"
            f"\n原始错误：{type(exc).__name__}: {exc}"
        ) from exc

    # 某些 diffusers 版本提供 set_adapters，用它显式设置 LoRA 权重。
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(
                args.adapter_name,
                adapter_weights=args.lora_scale,
            )
            print(f"[Load] LoRA scale: {args.lora_scale}", flush=True)
            return
        except (TypeError, ValueError):
            pass

    # 旧版可选择 fuse_lora；只有 scale != 1 时才需要显式融合。
    if args.lora_scale != 1.0 and hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora(
                adapter_names=[args.adapter_name],
                lora_scale=args.lora_scale,
            )
            print(f"[Load] fused LoRA scale: {args.lora_scale}", flush=True)
        except TypeError:
            pipe.fuse_lora(lora_scale=args.lora_scale)


def pipeline_supports(pipe: QwenImageEditPlusPipeline, parameter: str) -> bool:
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return False
    return parameter in signature.parameters


@torch.inference_mode()
def generate_one(
    pipe: QwenImageEditPlusPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    out_width: int,
    out_height: int,
    steps: int,
    true_cfg_scale: float,
    seed: int,
    max_sequence_length: int,
) -> Image.Image:
    # 使用 CPU generator，便于 device_map / CPU offload 场景下保持可复现。
    generator = torch.Generator(device="cpu").manual_seed(seed)

    call_kwargs = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": out_height,
        "width": out_width,
        "num_inference_steps": steps,
        "generator": generator,
        "output_type": "pil",
    }

    if pipeline_supports(pipe, "true_cfg_scale"):
        call_kwargs["true_cfg_scale"] = true_cfg_scale
    elif pipeline_supports(pipe, "guidance_scale"):
        # 兼容较旧或经过修改的 Pipeline；Qwen-Image-Edit-Plus 新版应优先使用 true_cfg_scale。
        call_kwargs["guidance_scale"] = true_cfg_scale
    else:
        raise RuntimeError(
            "当前 QwenImageEditPlusPipeline.__call__ 同时不包含 "
            "'true_cfg_scale' 和 'guidance_scale'，请检查 diffusers 版本。"
        )

    if pipeline_supports(pipe, "max_sequence_length"):
        call_kwargs["max_sequence_length"] = max_sequence_length

    output = pipe(**call_kwargs)
    if not hasattr(output, "images") or not output.images:
        raise RuntimeError("Pipeline 没有返回有效的 images。")
    return output.images[0]


def fit_panel(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    return ImageOps.pad(
        image,
        target_size,
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )


def make_comparison(
    input_image: Image.Image,
    base_image: Image.Image,
    lora_image: Image.Image,
) -> Image.Image:
    panel_width = max(base_image.width, lora_image.width)
    panel_height = max(base_image.height, lora_image.height)
    label_height = 44

    panels = [
        ("Input", fit_panel(input_image, (panel_width, panel_height))),
        ("Base", fit_panel(base_image, (panel_width, panel_height))),
        ("LoRA", fit_panel(lora_image, (panel_width, panel_height))),
    ]

    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, panel) in enumerate(panels):
        left = index * panel_width
        canvas.paste(panel, (left, label_height))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (left + (panel_width - text_width) // 2, 14),
            label,
            fill="black",
            font=font,
        )

    return canvas


def safe_output_name(sample_id: str, index: int, candidate_index: int) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in sample_id
    ).strip("._")
    if not cleaned:
        cleaned = f"sample_{index:06d}"
    return f"{index:06d}_{cleaned}_candidate_{candidate_index:02d}.png"


def ensure_dirs(output_dir: Path, mode: str, save_comparison: bool) -> dict[str, Path]:
    dirs = {"root": output_dir}
    if mode in ("base", "both"):
        dirs["base"] = output_dir / "base"
    if mode in ("lora", "both"):
        dirs["lora"] = output_dir / "lora"
    if mode == "both" and save_comparison:
        dirs["comparison"] = output_dir / "comparison"

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_metadata(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def should_skip(path: Path, overwrite: bool) -> bool:
    return path.exists() and not overwrite


def run() -> None:
    args = parse_args()

    if args.num_candidates < 1:
        raise ValueError("--num_candidates must be at least 1.")

    if not torch.cuda.is_available() and (
        args.device.startswith("cuda") or args.device_map is not None or args.cpu_offload
    ):
        raise RuntimeError("当前 Python 环境没有可用 CUDA。")

    if args.mode in ("lora", "both") and args.lora_path is None:
        raise ValueError("mode=lora/both 时必须传入 --lora_path。")

    samples = build_samples(args)
    if not samples:
        raise ValueError("没有读取到任何待处理样本。")

    output_dir = args.output_dir.expanduser().resolve()
    dirs = ensure_dirs(output_dir, args.mode, args.save_comparison)
    metadata_path = output_dir / "metadata.jsonl"

    print(f"[Info] samples: {len(samples)}", flush=True)
    print(f"[Info] output: {output_dir}", flush=True)
    print(
        f"[Info] steps={args.steps}, true_cfg_scale={args.true_cfg_scale}, "
        f"seed={args.seed}, candidates_per_image={args.num_candidates}, "
        f"resolution={args.resolution}, auto_resize={args.auto_resize}",
        flush=True,
    )

    pipe = load_pipeline(args)

    candidate_jobs = [
        (sample_index, candidate_index, sample)
        for sample_index, sample in enumerate(samples)
        for candidate_index in range(args.num_candidates)
    ]

    # mode=base/both 时，先在未加载 LoRA 的状态下生成 Base。
    base_outputs: dict[str, Path] = {}
    if args.mode in ("base", "both"):
        for job_index, (index, candidate_index, sample) in enumerate(candidate_jobs):
            if not sample.image_path.is_file():
                raise FileNotFoundError(f"输入图片不存在：{sample.image_path}")

            output_name = safe_output_name(sample.sample_id, index, candidate_index)
            output_path = dirs["base"] / output_name
            base_outputs[output_name] = output_path

            if should_skip(output_path, args.overwrite):
                print(f"[Skip][Base] {output_path}", flush=True)
                continue

            image = Image.open(sample.image_path).convert("RGB")
            source_width, source_height = image.size
            generation_image, out_width, out_height, crop_box = prepare_generation_image(
                image=image,
                preserve_resolution=args.preserve_resolution,
                auto_resize=args.auto_resize,
                resolution=args.resolution,
                height=args.height,
                width=args.width,
            )
            sample_seed = args.seed + candidate_index
            if args.increment_seed:
                sample_seed += index * args.num_candidates

            print(
                f"[Base] {job_index + 1}/{len(candidate_jobs)} "
                f"{sample.image_path.name} -> {out_width}x{out_height}, seed={sample_seed}",
                flush=True,
            )
            result = generate_one(
                pipe=pipe,
                image=generation_image,
                prompt=sample.prompt,
                negative_prompt=args.negative_prompt,
                out_width=out_width,
                out_height=out_height,
                steps=args.steps,
                true_cfg_scale=args.true_cfg_scale,
                seed=sample_seed,
                max_sequence_length=args.max_sequence_length,
            )
            if crop_box is not None:
                result = result.crop(crop_box)
            result.save(output_path)

            write_metadata(
                metadata_path,
                {
                    "mode": "base",
                    "index": index,
                    "group_id": sample.sample_id,
                    "candidate_index": candidate_index,
                    "num_candidates": args.num_candidates,
                    "sample_id": sample.sample_id,
                    "input_image": str(sample.image_path.resolve()),
                    "output_image": str(output_path),
                    "prompt": sample.prompt,
                    "negative_prompt": args.negative_prompt,
                    "seed": sample_seed,
                    "width": result.width,
                    "height": result.height,
                    "source_width": source_width,
                    "source_height": source_height,
                    "generation_width": out_width,
                    "generation_height": out_height,
                    "preserve_resolution": args.preserve_resolution,
                    "steps": args.steps,
                    "true_cfg_scale": args.true_cfg_scale,
                    "model_path": str(args.model_path.expanduser().resolve()),
                    "lora_path": None,
                    "teacher_checkpoint": str(args.model_path.expanduser().resolve()),
                    "bank_version": args.bank_version,
                    "reward_vector": None,
                    "confidence": None,
                    "selected_as_pseudo_gt": False,
                },
            )

    # mode=lora/both：在同一个 Base Pipeline 上加载 LoRA 后，用完全相同参数重新生成。
    if args.mode in ("lora", "both"):
        load_lora(pipe, args)

        for job_index, (index, candidate_index, sample) in enumerate(candidate_jobs):
            if not sample.image_path.is_file():
                raise FileNotFoundError(f"输入图片不存在：{sample.image_path}")

            output_name = safe_output_name(sample.sample_id, index, candidate_index)
            output_path = dirs["lora"] / output_name

            if should_skip(output_path, args.overwrite):
                print(f"[Skip][LoRA] {output_path}", flush=True)
            else:
                image = Image.open(sample.image_path).convert("RGB")
                source_width, source_height = image.size
                generation_image, out_width, out_height, crop_box = prepare_generation_image(
                    image=image,
                    preserve_resolution=args.preserve_resolution,
                    auto_resize=args.auto_resize,
                    resolution=args.resolution,
                    height=args.height,
                    width=args.width,
                )
                sample_seed = args.seed + candidate_index
                if args.increment_seed:
                    sample_seed += index * args.num_candidates

                print(
                    f"[LoRA] {job_index + 1}/{len(candidate_jobs)} "
                    f"{sample.image_path.name} -> {out_width}x{out_height}, seed={sample_seed}",
                    flush=True,
                )
                result = generate_one(
                    pipe=pipe,
                    image=generation_image,
                    prompt=sample.prompt,
                    negative_prompt=args.negative_prompt,
                    out_width=out_width,
                    out_height=out_height,
                    steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    seed=sample_seed,
                    max_sequence_length=args.max_sequence_length,
                )
                if crop_box is not None:
                    result = result.crop(crop_box)
                result.save(output_path)

                write_metadata(
                    metadata_path,
                    {
                        "mode": "lora",
                        "index": index,
                        "group_id": sample.sample_id,
                        "candidate_index": candidate_index,
                        "num_candidates": args.num_candidates,
                        "sample_id": sample.sample_id,
                        "input_image": str(sample.image_path.resolve()),
                        "output_image": str(output_path),
                        "prompt": sample.prompt,
                        "negative_prompt": args.negative_prompt,
                        "seed": sample_seed,
                        "width": result.width,
                        "height": result.height,
                        "source_width": source_width,
                        "source_height": source_height,
                        "generation_width": out_width,
                        "generation_height": out_height,
                        "preserve_resolution": args.preserve_resolution,
                        "steps": args.steps,
                        "true_cfg_scale": args.true_cfg_scale,
                        "model_path": str(args.model_path.expanduser().resolve()),
                        "lora_path": str(args.lora_path.expanduser().resolve()),
                        "teacher_checkpoint": str(args.lora_path.expanduser().resolve()),
                        "adapter_name": args.adapter_name,
                        "lora_scale": args.lora_scale,
                        "bank_version": args.bank_version,
                        "reward_vector": None,
                        "confidence": None,
                        "selected_as_pseudo_gt": False,
                    },
                )

            if args.mode == "both" and args.save_comparison:
                base_path = base_outputs[output_name]
                lora_path = output_path
                comparison_path = dirs["comparison"] / output_name

                if (
                    base_path.is_file()
                    and lora_path.is_file()
                    and not should_skip(comparison_path, args.overwrite)
                ):
                    input_image = Image.open(sample.image_path).convert("RGB")
                    base_image = Image.open(base_path).convert("RGB")
                    lora_image = Image.open(lora_path).convert("RGB")
                    comparison = make_comparison(
                        input_image=input_image,
                        base_image=base_image,
                        lora_image=lora_image,
                    )
                    comparison.save(comparison_path)

    print(f"[Done] results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[Error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
