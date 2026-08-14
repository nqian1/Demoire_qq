import os
import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)


# =====================================================
# Path
# =====================================================

MODEL_PATH = (
    "/home/wx1522744/zhaoqianqian/models/Qwen3.6-27B"
)

PROJECT_ROOT = (
    "/data_mount_162/zhaoqianqian/code/Flow-Factory"
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "moire_final"
)

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "images"
)

TRAIN_JSON = os.path.join(
    DATASET_ROOT,
    "train.jsonl"
)

TEST_JSON = os.path.join(
    DATASET_ROOT,
    "test.jsonl"
)

OUTPUT_DIR = os.path.join(
    DATASET_ROOT,
    "prompts"
)

SYSTEM_PROMPT_DIR = os.path.join(
    PROJECT_ROOT,
    "tools",
    "datasets_process",
    "system_prompts"
)


# =====================================================
# Load System Prompt
# =====================================================

def load_system_prompt(prompt_type):

    path = os.path.join(
        SYSTEM_PROMPT_DIR,
        f"system_prompt_{prompt_type}.txt"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 system prompt: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read().strip()


# =====================================================
# Load Model
# =====================================================

def load_model():

    print("Loading Qwen3.6-27B...")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    return model, processor


# =====================================================
# Inference
# =====================================================

def rewrite_prompt(
    model,
    processor,
    system_prompt,
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": "用户原始编辑需求：去摩尔纹"
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(
        model.device
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )

    output_ids = [
        out[len(inp):]
        for inp, out in zip(
            inputs.input_ids,
            outputs
        )
    ]

    result = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return result.strip()


# =====================================================
# Load JSONL
# =====================================================

def load_jsonl(path):

    data = []

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 JSONL: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            data.append(
                json.loads(line)
            )

    return data


# =====================================================
# Read Finished Samples
# =====================================================

def load_finished_images(output_file):

    finished = set()

    if not os.path.exists(output_file):
        return finished

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line.startswith("IMAGE:"):

                image_name = (
                    line
                    .replace("IMAGE:", "", 1)
                    .strip()
                )

                if image_name:
                    finished.add(
                        image_name
                    )

    return finished


# =====================================================
# Build Record
# =====================================================

def build_record(
    image_name,
    source,
    split,
    image_type,
    result
):

    return (
        "\n"
        "============================================================\n"
        f"IMAGE: {image_name}\n"
        f"SOURCE: {source}\n"
        f"SPLIT: {split}\n"
        f"TYPE: {image_type}\n"
        "\n"
        "PROMPT:\n"
        f"{result}\n"
        "============================================================\n"
    )


# =====================================================
# Process One Split
# =====================================================

def process_split(
    model,
    processor,
    system_prompt,
    prompt_type,
    split,
    json_path,
    max_samples=None
):

    print("\n" + "=" * 60)
    print(
        f"Processing Prompt {prompt_type} / {split}"
    )
    print("=" * 60)

    dataset = load_jsonl(
        json_path
    )

    if max_samples is not None:
        dataset = dataset[:max_samples]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"prompt_{prompt_type}_{split}.txt"
    )

    finished = load_finished_images(
        output_file
    )

    print(
        f"Total samples : {len(dataset)}"
    )

    print(
        f"Already done  : {len(finished)}"
    )

    print(
        f"Output file   : {output_file}"
    )

    success_count = 0
    error_count = 0
    missing_count = 0
    skip_count = 0

    with open(
        output_file,
        "a",
        encoding="utf-8"
    ) as f:

        for item in tqdm(
            dataset,
            desc=f"{prompt_type}-{split}"
        ):

            image_name = item["image"]

            if image_name in finished:

                skip_count += 1
                continue

            image_path = os.path.join(
                IMAGE_DIR,
                image_name
            )

            if not os.path.exists(
                image_path
            ):

                print(
                    f"\n[Missing] {image_path}"
                )

                missing_count += 1
                continue

            source = item.get(
                "source",
                image_name
            )

            image_type = item.get(
                "type",
                "unknown"
            )

            try:

                result = rewrite_prompt(
                    model=model,
                    processor=processor,
                    system_prompt=system_prompt,
                    image_path=image_path
                )

                record = build_record(
                    image_name=image_name,
                    source=source,
                    split=split,
                    image_type=image_type,
                    result=result
                )

                f.write(record)
                f.flush()

                success_count += 1

            except Exception as e:

                error_count += 1

                print(
                    f"\n[ERROR] "
                    f"{image_name}: "
                    f"{type(e).__name__}: {e}"
                )

    print("\nSplit finished.")

    print(
        f"Success : {success_count}"
    )

    print(
        f"Skipped : {skip_count}"
    )

    print(
        f"Missing : {missing_count}"
    )

    print(
        f"Errors  : {error_count}"
    )


# =====================================================
# Main
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "使用 Qwen3.6-27B 为摩尔纹数据生成 "
            "A/B/C/D 四种编辑 prompt"
        )
    )

    parser.add_argument(
        "--prompt_type",
        required=True,
        choices=[
            "A",
            "B",
            "C",
            "D"
        ],
        help="选择当前GPU负责的prompt类型。"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "每个split最多处理多少张。"
            "默认None表示全部。"
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "test",
            "both"
        ],
        default="both",
        help=(
            "处理train、test或两者。"
            "默认both。"
        )
    )

    args = parser.parse_args()

    prompt_type = args.prompt_type

    print(
        f"Current Prompt Type: {prompt_type}"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ================================================
    # Load System Prompt
    # ================================================

    system_prompt = load_system_prompt(
        prompt_type
    )

    print(
        f"System prompt loaded: "
        f"system_prompt_{prompt_type}.txt"
    )

    # ================================================
    # Load Model
    # ================================================

    model, processor = load_model()

    # ================================================
    # Process
    # ================================================

    if args.split in (
        "train",
        "both"
    ):

        process_split(
            model=model,
            processor=processor,
            system_prompt=system_prompt,
            prompt_type=prompt_type,
            split="train",
            json_path=TRAIN_JSON,
            max_samples=args.max_samples
        )

    if args.split in (
        "test",
        "both"
    ):

        process_split(
            model=model,
            processor=processor,
            system_prompt=system_prompt,
            prompt_type=prompt_type,
            split="test",
            json_path=TEST_JSON,
            max_samples=args.max_samples
        )

    print("\n" + "=" * 60)

    print(
        f"Finished Prompt {prompt_type}"
    )

    print(
        f"Results saved in: {OUTPUT_DIR}"
    )

    print("=" * 60)


# =====================================================
# Entry
# =====================================================

if __name__ == "__main__":
 
    main()
