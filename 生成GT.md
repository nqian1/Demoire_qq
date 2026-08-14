# 八张卡分别生成 A/B/C/D 伪GT候选命令

## 运行方式

以下方案不合并JSONL，每张GPU单独读取一个文件：

| 物理GPU | Prompt类型 | 数据集 |
|---:|---|---|
| 0 | A | train |
| 1 | A | test |
| 2 | B | train |
| 3 | B | test |
| 4 | C | train |
| 5 | C | test |
| 6 | D | train |
| 7 | D | test |

每个进程通过`CUDA_VISIBLE_DEVICES`只看到一张物理显卡，因此脚本内部统一使用`--device cuda:0`。这里的`cuda:0`指该进程可见的第一张卡，不是所有进程都占用物理0号卡。

## 第一步：进入项目并创建目录

```bash
cd /data_mount_162/zhaoqianqian/code/Flow-Factory
mkdir -p logs/pseudo_gt
mkdir -p outputs/pseudo_gt_candidates
```

如果服务器上的项目实际不在该路径，请把`cd`后的路径换成服务器真实路径。

## 第二步：启动八个后台任务

### GPU 0：Prompt A train

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/infer_qwen_demoire.py \
  --mode base \
  --jsonl dataset/moire_final/prompt_A_train.jsonl \
  --image_root dataset/moire_final/images \
  --output_dir outputs/pseudo_gt_candidates/A_train \
  --device cuda:0 \
  > logs/pseudo_gt/A_train.log 2>&1 &
```

### GPU 1：Prompt A test

```bash
CUDA_VISIBLE_DEVICES=1 nohup python tools/infer_qwen_demoire.py \
  --mode base \
  --jsonl dataset/moire_final/prompt_A_test.jsonl \
  --image_root dataset/moire_final/images \
  --output_dir outputs/pseudo_gt_candidates/A_test \
  --device cuda:0 \
  > logs/pseudo_gt/A_test.log 2>&1 &
```

### GPU 2：Prompt B train

```bash
CUDA_VISIBLE_DEVICES=2 nohup python tools/infer_qwen_demoire.py \
  --mode base \
  --jsonl dataset/moire_final/prompt_B_train.jsonl \
  --image_root dataset/moire_final/images \
  --output_dir outputs/pseudo_gt_candidates/B_train \
  --device cuda:0 \
  > logs/pseudo_gt/B_train.log 2>&1 &
```

### GPU 3：Prompt B test

```bash
CUDA_VISIBLE_DEVICES=3 nohup python tools/infer_qwen_demoire.py \
  --mode base \
  --jsonl dataset/moire_final/prompt_B_test.jsonl \
  --image_root dataset/moire_final/images \
  --output_dir outputs/pseudo_gt_candidates/B_test \
  --device cuda:0 \
  > logs/pseudo_gt/B_test.log 2>&1 &
```

### GPU 4：Prompt C train

```bash
CUDA_VISIBLE_DEVICES=4 nohup python tools/infer_qwen_demoire.py \
  --mode base \
  --jsonl dataset/moire_final/prompt_C_train.jsonl \
  --image_root dataset/moire_final/images \
  --output_dir outputs/pseudo_gt_candidates/C_train \
  --device cuda:0 \
  > logs/pseudo_gt/C_train.log 2>&1 &
```

### GPU 5：Prompt C test

```bash
CUDA_VISIBLE_DEVICES=5 nohup python tools/infer_qwen_demoire.py \
  --mode base \
  --jsonl dataset/moire_final/prompt_C_test.jsonl \
  --image_root dataset/moire_final/images \
  --output_dir outputs/pseudo_gt_candidates/C_test \
  --device cuda:0 \
  > logs/pseudo_gt/C_test.log 2>&1 &
```

### GPU 6：Prompt D train

```bash
CUDA_VISIBLE_DEVICES=6 nohup python tools/infer_qwen_demoire.py \
  --mode base \
  --jsonl dataset/moire_final/prompt_D_train.jsonl \
  --image_root dataset/moire_final/images \
  --output_dir outputs/pseudo_gt_candidates/D_train \
  --device cuda:0 \
  > logs/pseudo_gt/D_train.log 2>&1 &
```

### GPU 7：Prompt D test

```bash
CUDA_VISIBLE_DEVICES=7 nohup python tools/infer_qwen_demoire.py \
  --mode base \
  --jsonl dataset/moire_final/prompt_D_test.jsonl \
  --image_root dataset/moire_final/images \
  --output_dir outputs/pseudo_gt_candidates/D_test \
  --device cuda:0 \
  > logs/pseudo_gt/D_test.log 2>&1 &
```

## 第三步：查看运行状态

查看GPU：

```bash
watch -n 2 nvidia-smi
```

查看八个Python进程：

```bash
ps -ef | grep infer_qwen_demoire.py | grep -v grep
```

查看某个任务的实时日志，例如A训练集：

```bash
tail -f logs/pseudo_gt/A_train.log
```

快速查看所有日志末尾：

```bash
tail -n 20 logs/pseudo_gt/*.log
```

## 输出位置

每组结果完全隔离：

```text
outputs/pseudo_gt_candidates/
├── A_train/base/
├── A_test/base/
├── B_train/base/
├── B_test/base/
├── C_train/base/
├── C_test/base/
├── D_train/base/
└── D_test/base/
```

每个组自己的`metadata.jsonl`位于对应组目录下，例如：

```text
outputs/pseudo_gt_candidates/A_train/metadata.jsonl
outputs/pseudo_gt_candidates/A_test/metadata.jsonl
```

## 中断后继续

脚本默认跳过已经存在的输出图片。因此进程中断后，重新执行同一条命令即可继续，不要增加`--overwrite`。

只有明确想重新生成并覆盖已有图片时，才使用：

```text
--overwrite
```

## 显存不足时

如果单张GPU放不下模型，可在相应命令中添加：

```text
--cpu_offload
```

使用`--cpu_offload`会明显降低速度，并占用较多内存。不要同时使用`--cpu_offload`和`--device_map`。

同时启动八份模型还会占用大量CPU内存。建议先启动GPU 0的一条命令，确认模型、显存、图片路径和输出正常，再启动其余七条。
