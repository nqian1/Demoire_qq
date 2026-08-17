#!/usr/bin/env python3
"""Blind extraction + evidence-based adjudication for ComplexBench images."""

import argparse
import base64
import io
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "ComplexBench_1500_final_v3"


def arguments():
    parser = argparse.ArgumentParser(description="Gemini 两阶段盲提取与精准描述审校")
    parser.add_argument("--input", type=Path, default=DATA_ROOT / "ComplexBench_instruction_v3.jsonl")
    parser.add_argument("--images", type=Path, default=DATA_ROOT / "images")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "ComplexBench_instruction_v3_gemini_checked.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "extract_cache")
    parser.add_argument("--extract-prompt", type=Path, default=HERE / "prompt_extract.txt")
    parser.add_argument("--adjudicate-prompt", type=Path, default=HERE / "prompt_adjudicate.txt")
    parser.add_argument("--base-url", default=os.getenv("YIBU_BASE_URL", "https://yibuapi.com/v1"))
    parser.add_argument("--api-key-env", default="YIBU_API_KEY")
    parser.add_argument("--ca-bundle", type=Path, help="公司代理 CA 证书 PEM 文件（推荐）")
    parser.add_argument("--insecure", action="store_true", help="跳过 HTTPS 证书校验，仅用于可信代理环境下的临时测试")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--stage", choices=("all", "extract", "adjudicate"), default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--tile-trigger", type=int, default=1800, help="长边达到该像素时启用切片")
    parser.add_argument("--tile-size", type=int, default=1400)
    parser.add_argument("--tile-overlap", type=int, default=140)
    parser.add_argument("--max-tiles", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=2200)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    result = []
    with path.open("r", encoding="utf-8-sig") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {number} 行 JSON 错误: {exc}") from exc
    return result


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def safe_id(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))


def resize_within(image, max_side):
    copy = image.copy()
    copy.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return copy


def tile_boxes(width, height, tile_size, overlap, max_tiles):
    if max(width, height) < tile_size:
        return []
    preferred_step = tile_size - overlap
    if preferred_step <= 0:
        raise ValueError("tile-overlap 必须小于 tile-size")

    def starts(length):
        last = max(0, length - tile_size)
        natural = list(range(0, last + 1, preferred_step))
        if not natural or natural[-1] != last:
            natural.append(last)
        if len(natural) <= max_tiles:
            return natural
        if max_tiles == 1:
            return [0]
        # 图片过长时在全长上均匀取块，确保头尾都被覆盖。
        return sorted({round(last * index / (max_tiles - 1)) for index in range(max_tiles)})

    if height >= width:
        return [(0, top, width, min(height, top + tile_size)) for top in starts(height)]
    return [(left, 0, min(width, left + tile_size), height) for left in starts(width)]


def jpeg_data_url(image, quality):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_parts(path, args):
    with Image.open(path) as source:
        source = source.convert("RGB")
        parts = [("完整图片", resize_within(source, args.max_image_side))]
        if max(source.size) >= args.tile_trigger:
            for index, box in enumerate(tile_boxes(*source.size, args.tile_size, args.tile_overlap, args.max_tiles), 1):
                parts.append((f"放大分块 {index}，原图坐标 {box}", source.crop(box)))
    return [(label, jpeg_data_url(image, args.jpeg_quality)) for label, image in parts]


def multimodal_content(intro, parts):
    content = [{"type": "text", "text": intro}]
    for label, url in parts:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    return content


def extract_json(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型返回空内容")
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        left, right = cleaned.find("{"), cleaned.rfind("}")
        if left >= 0 and right > left:
            return json.loads(cleaned[left : right + 1])
        raise


def api_call(args, api_key, system_prompt, user_content):
    payload = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    if args.insecure:
        ssl_context = ssl._create_unverified_context()
    elif args.ca_bundle:
        ssl_context = ssl.create_default_context(cafile=str(args.ca_bundle))
    else:
        ssl_context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout, context=ssl_context) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"API 响应结构异常: {str(body)[:1200]}") from exc
    return extract_json(text), body.get("usage", {})


def retry_call(args, api_key, system_prompt, content, task_label):
    last_error = None
    for attempt in range(1, args.attempts + 1):
        tqdm.write(f"[{task_label}] API 请求 {attempt}/{args.attempts} 已发送，等待响应……")
        try:
            result = api_call(args, api_key, system_prompt, content)
            tqdm.write(f"[{task_label}] API 响应成功")
            return result
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            tqdm.write(f"[{task_label}] 第 {attempt} 次失败: {exc}")
            if attempt < args.attempts:
                wait_seconds = min(30, 2 ** (attempt - 1) + random.random())
                tqdm.write(f"[{task_label}] {wait_seconds:.1f} 秒后重试")
                time.sleep(wait_seconds)
    raise last_error


def validate_extraction(value):
    required = ("text_blocks", "visual_facts", "global_uncertainties")
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError("盲提取结果字段不完整")
    if not all(isinstance(value[key], list) for key in required):
        raise ValueError("盲提取三个字段都必须是数组")
    return value


def validate_final(value):
    required = ("decision", "needs_human_review", "review_reasons", "change_reasons",
                "text_blocks", "prompt_cn", "prompt_en")
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError("终审结果字段不完整")
    if value["decision"] not in ("PASS", "REVISE"):
        raise ValueError("decision 必须是 PASS 或 REVISE")
    if not isinstance(value["needs_human_review"], bool):
        raise ValueError("needs_human_review 必须是布尔值")
    for key in ("review_reasons", "change_reasons"):
        if not isinstance(value[key], list):
            raise ValueError(f"{key} 必须是数组")
    if not isinstance(value["text_blocks"], list):
        raise ValueError("text_blocks 必须是数组")
    block_keys = ("content", "language", "position", "typography", "role")
    for index, block in enumerate(value["text_blocks"], 1):
        if not isinstance(block, dict):
            raise ValueError(f"text_blocks[{index}] 必须是对象")
        for key in block_keys:
            if not isinstance(block.get(key), str) or not block[key].strip():
                raise ValueError(f"text_blocks[{index}].{key} 不能为空")
    for key in ("prompt_cn", "prompt_en"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{key} 不能为空")
    return value


def legacy_text_blocks(blocks):
    """Convert structured model output to the exact v3 text_blocks string format."""
    if not blocks:
        return "无可见文本"
    sections = []
    for index, block in enumerate(blocks, 1):
        sections.append(
            f"text_{index}:\n"
            f"内容: {block['content'].strip()}\n"
            f"语言: {block['language'].strip()}\n"
            f"位置: {block['position'].strip()}\n"
            f"字体/排版: {block['typography'].strip()}\n"
            f"作用: {block['role'].strip()}"
        )
    return "\n\n".join(sections)


def old_annotation(item):
    return {key: item.get(key, "") for key in ("text_blocks", "prompt_cn", "prompt_en")}


def adjudication_intro(item, extraction):
    labels = {key: item.get(key, "") for key in ("domain", "category", "attributes")}
    return (
        "请完成终审。以下盲提取与旧标注都只是待核对材料，最终以图片为准。\n\n"
        f"辅助标签：\n{json.dumps(labels, ensure_ascii=False, indent=2)}\n\n"
        f"独立盲提取：\n{json.dumps(extraction, ensure_ascii=False, indent=2)}\n\n"
        f"Qwen 旧标注：\n{json.dumps(old_annotation(item), ensure_ascii=False, indent=2)}"
    )


def load_done(path):
    if not path.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(path) if row.get("id") is not None}


def main():
    args = arguments()
    if not args.input.is_file() or not args.images.is_dir():
        raise FileNotFoundError("输入 JSONL 或图片目录不存在")
    if not args.extract_prompt.is_file() or not args.adjudicate_prompt.is_file():
        raise FileNotFoundError("提示词文件不存在")
    if args.start < 0 or (args.end != -1 and args.end < args.start):
        raise ValueError("start/end 参数无效")
    if args.attempts < 1 or args.limit < 0:
        raise ValueError("attempts 必须 >= 1，limit 必须 >= 0")
    if args.insecure and args.ca_bundle:
        raise ValueError("--insecure 与 --ca-bundle 不能同时使用")
    if args.ca_bundle and not args.ca_bundle.is_file():
        raise FileNotFoundError(f"CA 证书不存在: {args.ca_bundle}")

    rows = read_jsonl(args.input)[args.start : None if args.end == -1 else args.end]
    if args.ids:
        wanted = set(map(str, args.ids))
        rows = [row for row in rows if str(row.get("id")) in wanted]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("筛选后没有记录")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output.with_name(args.output.stem + "_audit.jsonl")
    failure_path = args.output.with_name(args.output.stem + "_failures.jsonl")
    extract_prompt = args.extract_prompt.read_text(encoding="utf-8").strip()
    adjudicate_prompt = args.adjudicate_prompt.read_text(encoding="utf-8").strip()

    if args.dry_run:
        item = rows[0]
        image_path = args.images / Path(str(item.get("image", ""))).name
        parts = image_parts(image_path, args)
        print(json.dumps({"id": item.get("id"), "image": str(image_path), "image_parts": len(parts),
                          "stage": args.stage, "model": args.model,
                          "endpoint": args.base_url.rstrip("/") + "/chat/completions"},
                         ensure_ascii=False, indent=2))
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"请设置环境变量 {args.api_key_env}，不要将密钥写入文件")
    if args.insecure:
        print("警告：已使用 --insecure，HTTPS 证书不会被验证。建议后续改用 --ca-bundle。")
    done = load_done(args.output)
    stats = {"extracted": 0, "PASS": 0, "REVISE": 0, "human_review": 0, "failed": 0, "skipped": 0}

    for item in tqdm(rows, desc=args.stage):
        item_id = str(item.get("id", ""))
        cache_path = args.cache_dir / f"{safe_id(item_id)}.json"
        try:
            if not item_id:
                raise ValueError("记录缺少 id")
            image_path = args.images / Path(str(item.get("image", ""))).name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            parts = image_parts(image_path, args)

            if cache_path.is_file():
                extraction = validate_extraction(json.loads(cache_path.read_text(encoding="utf-8"))["extraction"])
            elif args.stage == "adjudicate":
                raise FileNotFoundError(f"缺少盲提取缓存: {cache_path}")
            else:
                extraction, usage = retry_call(
                    args, api_key, extract_prompt,
                    multimodal_content("请独立提取图片文字与视觉事实。", parts),
                    f"{item_id} 盲提取",
                )
                extraction = validate_extraction(extraction)
                cache_path.write_text(json.dumps({"id": item_id, "image": item.get("image"),
                                                  "extraction": extraction, "usage": usage},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
                stats["extracted"] += 1

            if args.stage == "extract":
                continue
            if item_id in done:
                stats["skipped"] += 1
                continue

            final, usage = retry_call(
                args, api_key, adjudicate_prompt,
                multimodal_content(adjudication_intro(item, extraction), parts),
                f"{item_id} 终审描述",
            )
            final = validate_final(final)
            checked = dict(item)
            checked["text_blocks"] = legacy_text_blocks(final["text_blocks"])
            checked["prompt_cn"] = final["prompt_cn"]
            checked["prompt_en"] = final["prompt_en"]
            append_jsonl(args.output, checked)
            append_jsonl(audit_path, {"id": item_id, "decision": final["decision"],
                                      "needs_human_review": final["needs_human_review"],
                                      "review_reasons": final["review_reasons"],
                                      "change_reasons": final["change_reasons"], "usage": usage})
            stats[final["decision"]] += 1
            stats["human_review"] += int(final["needs_human_review"])
            done.add(item_id)
        except Exception as exc:
            append_jsonl(failure_path, {"id": item_id, "image": item.get("image", ""), "error": str(exc)})
            stats["failed"] += 1

    print(json.dumps(stats, ensure_ascii=False))
    print(f"结果: {args.output}\n审计: {audit_path}\n失败: {failure_path}\n盲提取缓存: {args.cache_dir}")


if __name__ == "__main__":
    main()
