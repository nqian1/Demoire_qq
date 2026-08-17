# Gemini 两阶段审校

本目录的流程先让 Gemini 在看不到 Qwen 标注的情况下独立识别图片，再让它对照图片、独立结果和旧标注进行裁决并生成中英文描述。长图会自动切成带重叠的放大分块。

## 运行

先在 API 平台轮换已经公开过的密钥，然后通过环境变量输入新密钥：

```bash
cd /home/wx1522744/zhaoqianqian/project/Complex_bench
read -rsp "YIBU API Key: " YIBU_API_KEY
export YIBU_API_KEY
```

先做 10 条测试：

```bash
python gemini_check/check.py --limit 10
```

如果服务器通过公司 HTTPS 代理访问外网，并提示自签名证书错误，推荐指定公司 CA：

```bash
python gemini_check/check.py --limit 10 --ca-bundle /path/to/company-ca.pem
```

暂时拿不到公司 CA 时，可以在确认代理可信后测试（会跳过 HTTPS 证书验证）：

```bash
python gemini_check/check.py --limit 1 --insecure --attempts 1 --timeout 60
```

确认结果后执行全量；已有盲提取缓存和最终记录会自动复用或跳过：

```bash
python gemini_check/check.py
```

也可以拆开运行：

```bash
python gemini_check/check.py --stage extract
python gemini_check/check.py --stage adjudicate
```

指定记录：

```bash
python gemini_check/check.py --ids 00000154_5090 00000848_9515
```

默认生成：

- `ComplexBench_instruction_v3_gemini_checked.jsonl`：保持原数据字段结构的最终结果。
- `*_audit.jsonl`：PASS/REVISE、修改原因及人工复核标记。
- `*_failures.jsonl`：失败记录，可再次运行重试。
- `gemini_check/extract_cache/*.json`：盲提取结果，避免第二阶段失败后重复付费。

建议优先人工检查 `needs_human_review=true`，以及包含日期、金额、地址、电话号码和专业术语的记录。

