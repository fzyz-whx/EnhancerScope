"""M3 主实验：LoRA 微调基因组语言模型（DNABERT-2 / HyenaDNA）。

用法:
  uv run --group ml python scripts/finetune_lora.py --model dnabert2 --seed 42 --epochs 3

设计（8GB 显存硬约束，见 docs/devlog.md M3 计划）：
- LoRA r=16 alpha=32 dropout=0.1，target 按模型配置（configs/lora.json）
- bf16 权重 + gradient checkpointing + 梯度累积 2（等效 batch 64）
- 防泄漏契约：只见 train split；valid（chr2R 左半）早停；test（右半）只碰一次
- 记录：显存峰值 / 训练时长 / 每 epoch 曲线（results/logs/lora_<model>_seed<seed>.json）

坑位预案（devlog M3 预判）：
- HyenaDNA remote code × transformers 4.57：einops 已装，其余跑了才知道
- target_modules 命名按模型不同：跑前先打印 named_modules 实测确认
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enhancerscope.data import TARGETS, load_split  # noqa: E402
from enhancerscope.metrics import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lora.csv"
LOGS = ROOT / "results" / "logs"
CONFIG = json.loads((ROOT / "configs" / "lora.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hidden_of(model, input_ids, attention_mask):
    """兼容自定义模型的三个坑（devlog M3 实录）：
    1. DNABERT-2 标准注意力回退路径返回 tuple(last_hidden_state, pooled) 而非 ModelOutput
    2. HyenaDNA 的 forward 不接受 attention_mask 关键字
    """
    try:
        raw = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        raw = model(input_ids=input_ids)
    return raw[0] if isinstance(raw, tuple) else raw.last_hidden_state


def append_rows(rows: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    header = not RESULTS.exists()
    df.to_csv(RESULTS, mode="a", header=header, index=False)
    print(f"[lora] 已追加 {len(rows)} 行到 {RESULTS}")


class PoolerHead(nn.Module):
    """mean pooling over unmasked tokens + Linear(H, 2)。LoRA 只挂 backbone。"""

    def __init__(self, hidden: int, out: int = 2) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, out)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)


@torch.no_grad()
def predict(
    model: nn.Module,
    pooler: nn.Module,
    x_ids: torch.Tensor,
    x_mask: torch.Tensor,
    batch: int = 64,
) -> np.ndarray:
    out = []
    for i in range(0, x_ids.shape[0], batch):
        h = hidden_of(
            model, input_ids=x_ids[i : i + batch], attention_mask=x_mask[i : i + batch]
        )
        out.append(pooler(h, x_mask[i : i + batch]).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONFIG["models"]), required=True)
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--epochs", type=int, default=CONFIG["train"]["max_epochs"])
    args = ap.parse_args()

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    mcfg = CONFIG["models"][args.model]
    local = mcfg.get("local")
    model_path = local if local and Path(local).exists() else mcfg["hub"]
    print(f"[lora:{args.model}] 来源={model_path} seed={seed} device={DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # DNABERT-2 自定义代码有 dtype 硬编码缺陷（标准注意力回退路径 Float vs BFloat16 混算会炸，
    # devlog M3 有实录）→ fp32 加载权重 + 训练循环 autocast(bf16) 承担混合精度
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(DEVICE)
    hidden_size = (
        getattr(model.config, "hidden_size", None) or model.config.d_model
    )  # HyenaDNA 用 d_model
    print(
        f"[lora:{args.model}] hidden={hidden_size} 参数量={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )

    # target_modules 实测确认：打印候选名
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    want = CONFIG["lora"]["target_modules"][args.model]
    missing = [w for w in want if w not in names]
    if missing:
        print(
            f"[lora:{args.model}] !! 配置的 target_modules {missing} 不在模型中，实测顶层名："
        )
        from collections import Counter

        print(
            "  ",
            Counter(
                n.split(".")[-1]
                for n, _ in model.named_modules()
                if any(c.isalpha() for c in n)
            ).most_common(25),
        )
        return 1

    lcfg = LoraConfig(
        r=CONFIG["lora"]["r"],
        lora_alpha=CONFIG["lora"]["lora_alpha"],
        lora_dropout=CONFIG["lora"]["lora_dropout"],
        target_modules=want,
        bias=CONFIG["lora"]["bias"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lcfg)
    try:
        model.gradient_checkpointing_enable()
        print("[lora] gradient checkpointing ✓")
    except Exception as e:  # 如实记录：remote code 模型可能不支持
        print(f"[lora] !! gradient checkpointing 不可用: {e}")
    model.print_trainable_parameters()

    pooler = PoolerHead(hidden_size).to(DEVICE)  # 头部全参训练（很小）

    # tokenize（一次性）
    print("[lora] tokenize 中...")
    enc = {}
    tok_cache = ROOT / "data" / "processed" / "tokenized"
    tok_cache.mkdir(parents=True, exist_ok=True)
    for s, df in (
        ("train", load_split("train")),
        ("valid", load_split("valid")),
        ("test", load_split("test")),
    ):
        cache = tok_cache / f"{args.model}_{s}.pt"
        if cache.exists():
            toks = torch.load(
                cache, weights_only=False
            )  # 本地可信缓存；torch>=2.6 默认 weights_only=True 会拒绝 BatchEncoding
            print(f"  {s}: 命中 tokenize 缓存 {cache.name}")
        else:
            toks = tokenizer(
                list(df["sequence"]),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            torch.save(toks, cache)
            print(f"  {s}: ids {tuple(toks['input_ids'].shape)}（已缓存）")
        enc[s] = (toks["input_ids"], toks["attention_mask"])

    ids = {s: enc[s][0].to(DEVICE) for s in enc}
    mask = {s: enc[s][1].to(DEVICE) for s in enc}
    y = {
        s: torch.tensor(
            df[list(TARGETS)].to_numpy(), dtype=torch.float32, device=DEVICE
        )
        for s, df in (
            ("train", load_split("train")),
            ("valid", load_split("valid")),
            ("test", load_split("test")),
        )
    }

    tcfg = CONFIG["train"]
    accum = tcfg["grad_accum"]
    params = [p for p in model.parameters() if p.requires_grad] + list(
        pooler.parameters()
    )
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = nn.MSELoss()
    scaler_dtype = torch.bfloat16

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    curve: list[dict] = []
    best_loss, best_state, bad = float("inf"), None, 0
    n = ids["train"].shape[0]

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        opt.zero_grad()
        for step, i in enumerate(range(0, n, tcfg["batch_size"])):
            idx = perm[i : i + tcfg["batch_size"]]
            with torch.autocast(
                device_type="cuda", dtype=scaler_dtype, enabled=DEVICE.type == "cuda"
            ):
                h = hidden_of(
                    model,
                    input_ids=ids["train"][idx],
                    attention_mask=mask["train"][idx],
                )
                pred = pooler(h, mask["train"][idx])
                loss = loss_fn(pred, y["train"][idx]) / accum
            loss.backward()
            if (step + 1) % accum == 0 or i + tcfg["batch_size"] >= n:
                opt.step()
                opt.zero_grad()
            total += loss.item() * accum * len(idx)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, ids["valid"].shape[0], 128):
                h = hidden_of(
                    model,
                    input_ids=ids["valid"][i : i + 128],
                    attention_mask=mask["valid"][i : i + 128],
                )
                vl += (
                    loss_fn(
                        pooler(h, mask["valid"][i : i + 128]), y["valid"][i : i + 128]
                    ).item()
                    * 128
                )
            valid_loss = vl / ids["valid"].shape[0]
        sched_step = opt.param_groups[0]["lr"]
        curve.append(
            {
                "epoch": epoch,
                "train_loss": total / n,
                "valid_loss": valid_loss,
                "lr": sched_step,
            }
        )
        print(
            f"[lora:{args.model}] epoch={epoch} train={total / n:.4f} valid={valid_loss:.4f} lr={sched_step:.2e}"
        )
        if valid_loss < best_loss - 1e-5:
            best_loss, bad = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_pooler = {
                k: v.detach().clone() for k, v in pooler.state_dict().items()
            }
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                print(f"[lora:{args.model}] 早停 @ epoch={epoch}")
                break

    elapsed = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"lora_{args.model}_seed{seed}.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": seed,
                "elapsed_s": elapsed,
                "peak_mem_gb": peak_mem,
                "curve": curve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[lora:{args.model}] 时长 {elapsed / 60:.1f}min | 显存峰值 {peak_mem:.2f}GB | 曲线已存"
    )

    model.load_state_dict(best_state)
    pooler.load_state_dict(best_pooler)
    # 持久化最优权重（LoRA adapter + pooler；~2.4MB/模型）——M4 可解释性与 M5 ONNX 导出的输入。
    # 按 goal 规则不入库（data/ 已 gitignore），用脚本 + 本次运行日志复现。
    ckpt_dir = ROOT / "data" / "models" / "finetuned"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"lora_{args.model}_seed{seed}.pt"
    torch.save(
        {
            "lora_state_dict": {k: v for k, v in best_state.items() if "lora_" in k},
            "pooler_state_dict": best_pooler,
            "model": args.model,
            "seed": seed,
            "best_valid_mse": best_loss,
        },
        ckpt,
    )
    print(f"[lora:{args.model}] 权重已存 {ckpt}")
    model.eval()
    rows: list[dict] = []
    for split, df in (("valid", load_split("valid")), ("test", load_split("test"))):
        with torch.no_grad():
            p_all = []
            for i in range(0, ids[split].shape[0], 64):
                h = hidden_of(
                    model,
                    input_ids=ids[split][i : i + 64],
                    attention_mask=mask[split][i : i + 64],
                )
                p_all.append(pooler(h, mask[split][i : i + 64]).float().cpu().numpy())
            preds = np.concatenate(p_all)
        for ti, task in enumerate(TARGETS):
            m = evaluate(df[task].to_numpy(), preds[:, ti])
            rows.append(
                {
                    "model": f"lora_{args.model}",
                    "task": task,
                    "split": split,
                    "seed": seed,
                    **m,
                    "notes": f"r={CONFIG['lora']['r']} alpha={CONFIG['lora']['lora_alpha']} lr={tcfg['lr']} ep={epoch + 1}; 时长{elapsed / 60:.1f}min 显存{peak_mem:.2f}GB",
                }
            )
            print(f"[lora:{args.model}] {task} {split}: {m}")
    append_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
