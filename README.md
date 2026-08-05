# Qwen Demoire Pseudo-GT Generation Runbook

This runbook implements Phase A of `demoire_distilation.md`:

1. Generate four stochastic Qwen-Image-Edit candidates for every moire input.
2. Apply conservative hard constraints and heuristic ranking.
3. Export high-confidence pseudo-GT pairs.
4. Send rejected or ambiguous groups to manual review.

The automatic selector is a bootstrap filter. Human or VLM auditing is still
required before the selected bank is used to train the student model.

## 1. Activate the Flow-Factory environment

```bash
cd /path/to/Qwen_score/Flow-Factory
conda activate flow_factory
```

Replace the environment name if Flow-Factory is installed in another Conda
environment.

## 2. Prepare the input JSONL

Each line must contain an image path and may contain a custom prompt:

```json
{"image":"000001.jpg","prompt":"Remove all moire patterns while preserving text, geometry, color, and genuine detail exactly."}
{"image":"000002.jpg"}
```

When `prompt` is omitted, the built-in demoire prompt is used. Relative image
paths are resolved against `--image_root` when it is provided; otherwise they
are resolved against the directory containing the JSONL file.

## 3. Smoke test with two inputs

Run a small test before generating the full bank:

```bash
python tools/infer_qwen_demoire_candidates.py \
  --mode base \
  --jsonl /path/to/demoire_train.jsonl \
  --image_root /path/to/demoire/images \
  --output_dir outputs/qwen_candidates_smoke \
  --model_path /data/ckpts/Qwen/Qwen-Image-Edit-2509 \
  --max_samples 2 \
  --num_candidates 4 \
  --bank_version qwen-base-v1 \
  --seed 42 \
  --increment_seed \
  --preserve_resolution \
  --steps 20 \
  --true_cfg_scale 4.0 \
  --resolution 1024
```

Expected result: two input groups, four candidates per group, eight generated
images in total, and one `metadata.jsonl` file.

## 4. Generate the full candidate bank

```bash
python tools/infer_qwen_demoire_candidates.py \
  --mode base \
  --jsonl /path/to/demoire_train.jsonl \
  --image_root /path/to/demoire/images \
  --output_dir outputs/qwen_candidates_v1 \
  --model_path /data/ckpts/Qwen/Qwen-Image-Edit-2509 \
  --num_candidates 4 \
  --bank_version qwen-base-v1 \
  --seed 42 \
  --increment_seed \
  --preserve_resolution \
  --steps 20 \
  --true_cfg_scale 4.0 \
  --resolution 1024
```

The candidate seed is calculated as:

```text
seed = base_seed + sample_index * num_candidates + candidate_index
```

With `--seed 42 --num_candidates 4 --increment_seed`, the first input uses
seeds 42--45, the second input uses 46--49, and so on. The seed is written to
every metadata record, so each result is reproducible.

With `--preserve_resolution`, an input that is not divisible by 32 is
edge-padded to the next multiple of 32 for Qwen inference and cropped back to
its exact original width and height before it is saved. In this mode,
`--resolution` does not resize the image. Full-resolution 4K inference may
require substantially more GPU memory; run the smoke test first.

The output layout is:

```text
outputs/qwen_candidates_v1/
|-- base/
|   |-- 000000_sample_candidate_00.png
|   |-- 000000_sample_candidate_01.png
|   |-- 000000_sample_candidate_02.png
|   `-- 000000_sample_candidate_03.png
`-- metadata.jsonl
```

Do not use `--overwrite` when resuming an interrupted run. Existing candidates
will be skipped. Use `--overwrite` only when intentionally regenerating the
same bank version.

## 5. Select high-confidence pseudo GT

```bash
python tools/select_demoire_pseudo_gt.py \
  --metadata outputs/qwen_candidates_v1/metadata.jsonl \
  --output_dir outputs/pseudo_gt_v1 \
  --mode base \
  --copy_rejected_best
```

The selector applies:

- low-frequency content preservation;
- global color preservation;
- edge and detail preservation;
- suppression of periodic spectral peaks detected in the input;
- hard constraints, Top-1/Top-2 margin, and confidence filtering.

The result layout is:

```text
outputs/pseudo_gt_v1/
|-- images/
|-- manual_review/
|-- candidate_scores.jsonl
|-- selected_pseudo_gt.jsonl
|-- rejected_groups.jsonl
`-- selection_summary.json
```

`selected_pseudo_gt.jsonl` is the paired-data manifest. Each accepted record
contains `input_image`, `pseudo_gt_image`, the selected candidate, its reward
vector, confidence, seed metadata, and teacher checkpoint.

## 6. Review selection quality

Check `selection_summary.json` first. Then manually inspect at least 300 groups,
including:

- accepted groups with the lowest confidence;
- rejected groups with the highest score;
- text-heavy images;
- faces, logos, and thin geometric structures;
- strong color moire and large-scale curved moire.

Reject candidates that remove real texture, alter text, shift geometry, change
identity, introduce false texture, or obtain a high score by blurring the image.

If selection is too strict, rerun into a new output directory with slightly
relaxed thresholds:

```bash
python tools/select_demoire_pseudo_gt.py \
  --metadata outputs/qwen_candidates_v1/metadata.jsonl \
  --output_dir outputs/pseudo_gt_v1_relaxed \
  --mode base \
  --min_confidence 0.58 \
  --min_margin 0.01 \
  --copy_rejected_best
```

Do not reduce the thresholds further until manual ranking confirms that the
automatic ordering agrees with human preference.

## 7. Seed behavior

Changing only the random seed is sufficient to request a different stochastic
candidate while keeping the input, prompt, model, resolution, steps, and CFG
fixed. It does not guarantee that candidates will be visually very different:
strong image conditioning may make several seeds converge to similar edits.

Use different seeds for candidate diversity, but keep all other parameters
fixed within a group. Do not vary CFG, resolution, or prompt inside one group,
because that makes reward ranking harder to interpret.

If four seeds are consistently too similar, first test eight candidates:

```bash
--num_candidates 8
```

Prompt variants should be treated as a separate ablation or stored explicitly
in metadata, rather than silently mixed into the same seed-only candidate bank.
