# A/B/C/D 八个 Prompt JSONL 截断记录重新生成命令

## 功能说明

以下命令会检查已有JSONL，只重新生成不合格记录：

- prompt少于30个字符；
- prompt没有以句号、问号或叹号结尾；
- 本次生成达到`max_new_tokens`，可能发生截断。

已经合格的记录会直接跳过，其原始JSONL行不会重新生成、不会重新序列化、不会修改内容或格式。

发现不合格记录时，脚本会先备份原文件。例如：

```text
prompt_A_test.jsonl.before_retry.bak
```

然后只从工作文件中移除不合格行，并为相应图片重新生成prompt。

## 运行前

进入服务器上的Flow-Factory项目根目录：

```bash
cd /data_mount_162/zhaoqianqian/code/Flow-Factory
```

将修改后的脚本上传并覆盖到：

```text
tools/dataset_process/generate_moire_prompt.py
```

## 八条运行命令

### 1. Prompt A 训练集

```bash
python tools/dataset_process/generate_moire_prompt.py \
  --prompt_type A \
  --split train \
  --max_attempts 3
```

### 2. Prompt A 测试集

```bash
python tools/dataset_process/generate_moire_prompt.py \
  --prompt_type A \
  --split test \
  --max_attempts 3
```

### 3. Prompt B 训练集

```bash
python tools/dataset_process/generate_moire_prompt.py \
  --prompt_type B \
  --split train \
  --max_attempts 3
```

### 4. Prompt B 测试集

```bash
python tools/dataset_process/generate_moire_prompt.py \
  --prompt_type B \
  --split test \
  --max_attempts 3
```

### 5. Prompt C 训练集

```bash
python tools/dataset_process/generate_moire_prompt.py \
  --prompt_type C \
  --split train \
  --max_attempts 3
```

### 6. Prompt C 测试集

```bash
python tools/dataset_process/generate_moire_prompt.py \
  --prompt_type C \
  --split test \
  --max_attempts 3
```

### 7. Prompt D 训练集

```bash
python tools/dataset_process/generate_moire_prompt.py \
  --prompt_type D \
  --split train \
  --max_attempts 3
```

### 8. Prompt D 测试集

```bash
python tools/dataset_process/generate_moire_prompt.py \
  --prompt_type D \
  --split test \
  --max_attempts 3
```

## 输出位置

结果仍写回原来的八个文件：

```text
dataset/moire_final/prompt_A_train.jsonl
dataset/moire_final/prompt_A_test.jsonl
dataset/moire_final/prompt_B_train.jsonl
dataset/moire_final/prompt_B_test.jsonl
dataset/moire_final/prompt_C_train.jsonl
dataset/moire_final/prompt_C_test.jsonl
dataset/moire_final/prompt_D_train.jsonl
dataset/moire_final/prompt_D_test.jsonl
```

连续三次仍生成失败的样本不会写入主JSONL，而会记录在对应的失败文件中，例如：

```text
dataset/moire_final/prompt_A_test_failures.jsonl
```

## 注意事项

八条命令应依次执行。每条命令都会重新加载一次Qwen模型，结束后显存会由该Python进程释放。如果A、B、C、D分别使用不同GPU并行运行，也要确保每个进程使用不同显卡，避免显存冲突。
