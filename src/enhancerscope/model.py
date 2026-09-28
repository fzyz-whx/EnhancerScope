"""微调模型加载与推理（M4 归因 / M5 ONNX 导出 / M6 MCP 工具共用入口）。

设计：训练脚本（scripts/finetune_lora.py）与本模块共用 `hidden_of` / `PoolerHead`，
避免"训练时的模型包装"与"推理时的模型包装"漂移（M3 的五个兼容坑只解一次）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
DATA_MODELS = ROOT / "data" / "models"
CONFIG = json.loads((ROOT / "configs" / "lora.json").read_text(encoding="utf-8"))


def hidden_of(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """自定义模型兼容层（devlog M3 第 2/9 坑）：
    1. DNABERT-2 标准注意力回退路径返回 tuple(last_hidden_state, pooled)；
    2. HyenaDNA 底层 forward 不接受 attention_mask，且 peft 包装器会透传该参数
       → 需下钻到未包装的底层模型（LoRA 模块已注入其子层）。
    """
    try:
        raw = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        inner = getattr(getattr(model, "base_model", model), "model", None) or model
        raw = inner(input_ids=input_ids)
    return raw[0] if isinstance(raw, tuple) else raw.last_hidden_state


class PoolerHead(nn.Module):
    """mean pooling over unmasked tokens + Linear(H, 2)（与训练时结构一致）。"""

    def __init__(self, hidden: int, out: int = 2) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, out)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)


def local_base_dir(model_name: str) -> Path:
    cfg = CONFIG["models"][model_name]
    local = cfg.get("local")
    if local and Path(ROOT / local).exists():
        return ROOT / local
    raise FileNotFoundError(
        f"本地基座模型不存在（{local}）——先跑 scripts/download_model.py（DNABERT-2）"
        f" 或按 devlog M3 的方式下载 {cfg['hub']}"
    )


def load_finetuned(
    model_name: str = "dnabert2",
    seed: int | None = None,
    device: str | torch.device | None = None,
):
    """加载基座 + LoRA adapter + pooler 头。

    seed=None 时自动选择 `best_valid_mse` 最低的检查点（M4 的"最优模型"定义）。
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    ckpt_dir = DATA_MODELS / "finetuned"
    if seed is None:
        cands = sorted(ckpt_dir.glob(f"lora_{model_name}_seed*.pt"))
        if not cands:
            raise FileNotFoundError(
                f"没有微调权重: {ckpt_dir}/lora_{model_name}_seed*.pt（先跑 finetune_lora.py）"
            )
        best = min(
            cands,
            key=lambda p: torch.load(p, weights_only=False, map_location="cpu")[
                "best_valid_mse"
            ],
        )
        seed = int(best.stem.split("seed")[-1])
    ckpt_path = ckpt_dir / f"lora_{model_name}_seed{seed}.pt"
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base = local_base_dir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(base), trust_remote_code=True)
    lcfg = CONFIG["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["lora_alpha"],
            lora_dropout=lcfg["lora_dropout"],
            target_modules=lcfg["target_modules"][model_name],
            bias=lcfg["bias"],
            task_type=TaskType.FEATURE_EXTRACTION,
        ),
    )
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    hidden = getattr(model.config, "hidden_size", None) or model.config.d_model
    pooler = PoolerHead(hidden)
    pooler.load_state_dict(ckpt["pooler_state_dict"])
    model.eval().to(dev)
    pooler.eval().to(dev)
    return model, pooler, tokenizer, dev


@torch.no_grad()
def predict(
    model: Any,
    pooler: PoolerHead,
    tokenizer: Any,
    sequences: list[str],
    device: torch.device,
    batch: int = 64,
) -> np.ndarray:
    """批量预测 → (N, 2) [Dev, Hk]（fp32 前向，避免自定义代码的 dtype 坑）。"""
    outs = []
    for i in range(0, len(sequences), batch):
        toks = tokenizer(
            list(sequences[i : i + batch]),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        ids = toks["input_ids"].to(device)
        mask = (
            toks["attention_mask"].to(device)
            if "attention_mask" in toks
            else torch.ones_like(ids)
        )
        h = hidden_of(model, ids, mask)
        outs.append(pooler(h, mask).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 2), dtype=np.float32)
