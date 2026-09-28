# EnhancerScope Webapp（浏览器端增强子活性预测）

在线演示（部署后）：`https://<user>.github.io/EnhancerScope/`

## 它做什么

粘贴一段 DNA（249bp）→ 浏览器本地推理（ONNX Runtime Web / WASM）→ 输出 Dev/Hk 活性预测，
并给出**逐碱基敏感度**（单碱基饱和突变 |Δ预测|，与 `docs/interpretability.md` 的 M4 归因方法同源）。

## 为什么页面用的是 CNN 而不是 LoRA DNABERT-2

如实记录的工程取舍（devlog M5）：

| 模型 | ONNX 体积 | 浏览器可用性 | 数值一致性（vs PyTorch） |
|---|---|---|---|
| LoRA DNABERT-2（最优模型） | 481MB（fp32） | ✗ 网页加载不现实；int8 量化被导出图/量化器的形状冲突卡住；fp16 在 CPU/WASM 无法加载 | max\|Δ\|=7e-6，Pearson 1.0（results/onnx_parity.json）|
| **DeepSTARR 风格 CNN** | **0.81MB** | ✓ WASM 实测 2–15ms/序列 | max\|Δ\|=9.5e-7，Pearson 1.0 |

CNN 也不是"替身"：它是本仓库 M2 基准里 **Dev 侧最优**的模型（test ρ 0.639）。
LoRA 模型的 ONNX 导出仍然保留（作为 Release 资产 + 一致性报告），只是不作为网页加载对象。

## 本地运行

```bash
cd webapp
npm install          # 国内可用 --registry=https://registry.npmmirror.com
npm run build        # 产出 dist/（含 cnn.onnx 与 ORT 的 WASM 运行时）
npm run preview      # 本地预览
```

模型文件由 `scripts/export_cnn_onnx.py` 生成（训练 + 导出 + 一致性闸门）。
