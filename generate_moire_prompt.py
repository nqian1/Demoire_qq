import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_PATH = "/home/wx1522744/zhaoqianqian/models/Qwen3.6-27B"
PROJECT_ROOT = Path("/data_mount_162/zhaoqianqian/code/Flow-Factory")
DATASET_ROOT = PROJECT_ROOT / "dataset" / "moire_final"
IMAGE_DIR = DATASET_ROOT / "images"
TRAIN_JSON = DATASET_ROOT / "train.jsonl"
TEST_JSON = DATASET_ROOT / "test.jsonl"
# 相对脚本定位，避免旧代码中 dataset_process/datasets_process 拼写不一致。
SYSTEM_PROMPT_DIR = Path(__file__).resolve().parent / "system_prompts"
END_PUNCTUATION = ("\u3002", "\uff01", "\uff1f", ".", "!", "?")


def load_system_prompt(prompt_type):
    path = SYSTEM_PROMPT_DIR / f"system_prompt_{prompt_type}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"未找到system prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_model(model_path):
    print(f"Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def apply_chat_template(processor, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return processor.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        return processor.apply_chat_template(
            messages, **kwargs, chat_template_kwargs={"enable_thinking": False}
        )


def clean_prompt(text):
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.I | re.S)
    text = text.strip()
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.I)
    text = re.sub(r"^(?:最终编辑指令|编辑指令|PROMPT)\s*[:：]\s*", "", text, flags=re.I)
    # 只去掉包裹整段的引号，不改动句子内部引号。
    if len(text) >= 2 and (text[0], text[-1]) in {
        ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")
    }:
        text = text[1:-1].strip()
    return text


def validate_prompt(prompt, generated_tokens=None, max_new_tokens=None, min_chars=30):
    prompt = clean_prompt(prompt)
    reasons = []
    if not prompt:
        reasons.append("输出为空")
    if len(prompt) < min_chars:
        reasons.append(f"长度小于{min_chars}字符")
    if prompt and not prompt.endswith(END_PUNCTUATION):
        reasons.append("未以完整句末标点结尾，疑似截断")
    if generated_tokens is not None and max_new_tokens is not None:
        if generated_tokens >= max_new_tokens:
            reasons.append("达到max_new_tokens，疑似截断")
    return prompt, reasons


def rewrite_prompt(model, processor, system_prompt, image_path, max_new_tokens, retry_reason=""):
    retry_instruction = ""
    if retry_reason:
        retry_instruction = (
            f"\n上一次生成不合格，原因：{retry_reason}。请重新生成完整的一段编辑指令，"
            "必须以句号、叹号或问号结尾，不要延续或解释上一次输出。"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": "用户原始编辑需求：去摩尔纹" + retry_instruction,
                },
            ],
        },
    ]
    text = apply_chat_template(processor, messages)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    output_ids = outputs[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return result, int(output_ids.shape[1])


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到JSONL: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}第{line_number}行JSON错误: {exc}") from exc
    return rows


def prepare_output_jsonl(output_file, min_chars):
    """原样保留合格行；仅备份并移除不合格行，使其能够自动重跑。"""
    output_file = Path(output_file)
    if not output_file.exists():
        return set(), 0
    valid_lines, valid_images, invalid_count = [], set(), 0
    with output_file.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                valid_lines.append(line)
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{output_file}第{line_number}行JSON错误，请先修复：{exc}"
                ) from exc
            image_name = str(item.get("image", ""))
            _, reasons = validate_prompt(
                str(item.get("prompt", "")), min_chars=min_chars
            )
            if image_name and not reasons:
                # 直接保留原始行，保证其他合格记录的内容和格式均不改动。
                valid_lines.append(line)
                valid_images.add(image_name)
            else:
                invalid_count += 1
    if invalid_count:
        backup = output_file.with_suffix(output_file.suffix + ".before_retry.bak")
        if not backup.exists():
            shutil.copy2(output_file, backup)
        temp = output_file.with_suffix(output_file.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as file:
            file.writelines(valid_lines)
        os.replace(temp, output_file)
        print(f"发现{invalid_count}条不合格记录，原文件已备份到: {backup}")
    return valid_images, invalid_count


def build_record(item, split, prompt):
    return {
        "image": item["image"],
        "source": item.get("source", item["image"]),
        "split": split,
        "type": item.get("type", "unknown"),
        "prompt": prompt,
    }


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def process_split(
    model, processor, system_prompt, prompt_type, split, json_path,
    max_samples, max_attempts, max_new_tokens, min_chars,
):
    dataset = load_jsonl(json_path)
    if max_samples is not None:
        dataset = dataset[:max_samples]
    output_file = DATASET_ROOT / f"prompt_{prompt_type}_{split}.jsonl"
    failure_file = DATASET_ROOT / f"prompt_{prompt_type}_{split}_failures.jsonl"
    finished, invalid_count = prepare_output_jsonl(output_file, min_chars)
    print(f"\nProcessing {prompt_type}/{split}")
    print(f"Total={len(dataset)}, valid finished={len(finished)}, retry invalid={invalid_count}")
    print(f"Output={output_file}")
    counts = {"success": 0, "skipped": 0, "missing": 0, "errors": 0}

    for item in tqdm(dataset, desc=f"{prompt_type}-{split}"):
        image_name = str(item.get("image", ""))
        if image_name in finished:
            counts["skipped"] += 1
            continue
        image_path = IMAGE_DIR / image_name
        if not image_name or not image_path.is_file():
            print(f"\n[Missing] {image_path}")
            counts["missing"] += 1
            continue
        attempts, retry_reason, accepted = [], "", None
        try:
            for attempt in range(1, max_attempts + 1):
                raw, token_count = rewrite_prompt(
                    model, processor, system_prompt, image_path,
                    max_new_tokens, retry_reason,
                )
                prompt, reasons = validate_prompt(
                    raw, token_count, max_new_tokens, min_chars
                )
                attempts.append({
                    "attempt": attempt,
                    "generated_tokens": token_count,
                    "reasons": reasons,
                    "raw_output": raw,
                })
                if not reasons:
                    accepted = prompt
                    break
                retry_reason = "；".join(reasons)
                print(f"\n[Retry {attempt}/{max_attempts}] {image_name}: {retry_reason}")
            if accepted is None:
                raise ValueError(f"连续{max_attempts}次输出不完整")
            append_jsonl(output_file, build_record(item, split, accepted))
            finished.add(image_name)
            counts["success"] += 1
        except Exception as exc:
            counts["errors"] += 1
            append_jsonl(failure_file, {
                "image": image_name,
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempts,
            })
            print(f"\n[ERROR] {image_name}: {exc}")
    print(f"Finished {prompt_type}/{split}: {counts}")


def main():
    parser = argparse.ArgumentParser(description="生成A/B/C/D去摩尔纹prompt并直接写JSONL")
    parser.add_argument("--prompt_type", required=True, choices=list("ABCD"))
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_chars", type=int, default=30)
    parser.add_argument("--model_path", default=MODEL_PATH)
    args = parser.parse_args()
    if args.max_attempts < 1 or args.max_new_tokens < 1 or args.min_chars < 1:
        parser.error("max_attempts、max_new_tokens和min_chars必须大于0")
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    system_prompt = load_system_prompt(args.prompt_type)
    model, processor = load_model(args.model_path)
    if args.split in ("train", "both"):
        process_split(
            model, processor, system_prompt, args.prompt_type, "train", TRAIN_JSON,
            args.max_samples, args.max_attempts, args.max_new_tokens, args.min_chars,
        )
    if args.split in ("test", "both"):
        process_split(
            model, processor, system_prompt, args.prompt_type, "test", TEST_JSON,
            args.max_samples, args.max_attempts, args.max_new_tokens, args.min_chars,
        )


if __name__ == "__main__":
    main()
