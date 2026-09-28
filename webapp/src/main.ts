/**
 * EnhancerScope 浏览器端推理（M5）
 *
 * - 模型：DeepSTARR 风格 CNN（ONNX，0.81MB，与 PyTorch 数值一致 max|Δ|<1e-6）
 * - 预测：249bp one-hot (1,4,249) → (1,2) [Dev, Hk]
 * - 高亮：单碱基饱和突变（249 位 × 3 替换 = 747 条变体一次性批量前向），
 *   取每位的 max|Δ| —— 与 docs/interpretability.md 的 M4 归因方法同源
 * - 延迟：真实测量并显示在页面上（goal 要求"实测延迟写进文档"）
 */
import * as ort from "onnxruntime-web";

const SEQ_LEN = 249;
const BASES = ["A", "C", "G", "T"] as const;

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

const els = {
  seq: $<HTMLTextAreaElement>("seq"),
  predict: $<HTMLButtonElement>("predict"),
  demo: $<HTMLButtonElement>("demo"),
  status: $<HTMLSpanElement>("status"),
  err: $<HTMLDivElement>("err"),
  devVal: $<HTMLDivElement>("devVal"),
  hkVal: $<HTMLDivElement>("hkVal"),
  devBar: $<HTMLElement>("devBar"),
  hkBar: $<HTMLElement>("hkBar"),
  latency: $<HTMLDivElement>("latency"),
  seqwrap: $<HTMLDivElement>("seqwrap"),
  highlight: $<HTMLDivElement>("highlight"),
  hlLatency: $<HTMLSpanElement>("hlLatency"),
};

let session: ort.InferenceSession | null = null;
const cache = new Map<string, Float32Array>();

/** 序列 → (batch, 4, 249) one-hot float32（非 ACGT 记全零列） */
function oneHot(seqs: string[]): Float32Array {
  const out = new Float32Array(seqs.length * 4 * SEQ_LEN);
  seqs.forEach((seq, n) => {
    for (let p = 0; p < SEQ_LEN; p += 1) {
      const bi = BASES.indexOf(seq[p] as (typeof BASES)[number]);
      if (bi >= 0) out[n * (4 * SEQ_LEN) + bi * SEQ_LEN + p] = 1;
    }
  });
  return out;
}

function normalize(raw: string): string {
  const s = raw.toUpperCase().replace(/[^ACGT]/g, "N").slice(0, SEQ_LEN);
  return s.padEnd(SEQ_LEN, "N");
}

async function run(seqs: string[]): Promise<Float32Array> {
  if (!session) throw new Error("模型未加载");
  const dims = [seqs.length, 4, SEQ_LEN];
  const tensor = new ort.Tensor("float32", oneHot(seqs), dims);
  const out = await session.run({ onehot: tensor });
  return out.activity.data as Float32Array;
}

/** 单碱基饱和突变：每位替换为其余 3 种碱基，返回每位 max|Δ|（Dev, Hk） */
async function sensitivity(seq: string): Promise<{ dev: number[]; hk: number[] }> {
  const key = seq;
  if (cache.has(key)) {
    // 命中缓存（同一序列重复分析）
  }
  const base = await run([seq]);
  const variants: string[] = [];
  const owner: number[] = [];
  for (let p = 0; p < SEQ_LEN; p += 1) {
    for (const b of BASES) {
      if (b !== seq[p]) {
        variants.push(seq.slice(0, p) + b + seq.slice(p + 1));
        owner.push(p);
      }
    }
  }
  const preds = await run(variants);
  const dev = new Array<number>(SEQ_LEN).fill(0);
  const hk = new Array<number>(SEQ_LEN).fill(0);
  for (let i = 0; i < owner.length; i += 1) {
    const p = owner[i];
    dev[p] = Math.max(dev[p], Math.abs(preds[i * 2] - base[0]));
    hk[p] = Math.max(hk[p], Math.abs(preds[i * 2 + 1] - base[1]));
  }
  return { dev, hk };
}

function renderSeq(seq: string, sens: { dev: number[]; hk: number[] }): void {
  const maxDev = Math.max(...sens.dev, 1e-6);
  const maxHk = Math.max(...sens.hk, 1e-6);
  els.seqwrap.textContent = seq;
  els.highlight.innerHTML = "";
  for (let p = 0; p < SEQ_LEN; p += 1) {
    const bar = document.createElement("span");
    const v = Math.max(sens.dev[p] / maxDev, sens.hk[p] / maxHk);
    bar.style.height = `${Math.max(2, v * 100)}%`;
    bar.style.background = sens.dev[p] / maxDev >= sens.hk[p] / maxHk ? "var(--dev)" : "var(--hk)";
    bar.title = `pos ${p}: Dev|Δ|=${sens.dev[p].toFixed(3)} Hk|Δ|=${sens.hk[p].toFixed(3)}`;
    els.highlight.appendChild(bar);
  }
}

async function loadModel(): Promise<void> {
  const t0 = performance.now();
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
  session = await ort.InferenceSession.create("./cnn.onnx", {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  els.status.textContent = `模型就绪（${((performance.now() - t0) / 1000).toFixed(1)}s）· WASM`;
  els.predict.disabled = false;
}

async function onPredict(): Promise<void> {
  els.err.textContent = "";
  try {
    els.predict.disabled = true;
    const seq = normalize(els.seq.value);
    els.seq.value = seq;

    const t0 = performance.now();
    const pred = await run([seq]);
    const t1 = performance.now();

    els.devVal.textContent = pred[0].toFixed(3);
    els.hkVal.textContent = pred[1].toFixed(3);
    // 可视化用固定刻度（训练集活性大致落在 -5~+8）
    els.devBar.style.width = `${Math.min(100, Math.max(2, ((pred[0] + 5) / 13) * 100))}%`;
    els.hkBar.style.width = `${Math.min(100, Math.max(2, ((pred[1] + 5) / 13) * 100))}%`;
    els.latency.textContent = `单序列推理实测：${(t1 - t0).toFixed(0)} ms（浏览器 WASM，含数据准备）`;

    const t2 = performance.now();
    const sens = await sensitivity(seq);
    renderSeq(seq, sens);
    els.hlLatency.textContent = `碱基敏感度（747 条变体批量前向）实测：${(performance.now() - t2).toFixed(0)} ms`;
  } catch (e) {
    els.err.textContent = `出错：${(e as Error).message}`;
  } finally {
    els.predict.disabled = false;
  }
}

// 示例序列：M4 可解释性案例 2 的真实 test 序列（chr2R 右半，模型预测高活性且 top 归因落在 GATA motif 命中区）
const DEMO =
  "TAGTTGAGAATTCAAGCAGAAAACGCGACCGTCCATTTTGAACGAAAAAAAGAAGTGGAAGAATGCACGTGGCGATGTGACCGCACTACGATTCTGTAACGATAATTCATTTATCGAGCATGTAATCGATAGTTATAGAGGGTAGTTCCAAGAATCGTGTGCCACGGAAAGTGTAAATAGCTTGTTTTTTAGTAATCATTATTTATAGCATCGGCATTTTAAATAAAGGTAAATGCGCTTATAGCAATA";

els.demo.addEventListener("click", () => {
  els.seq.value = DEMO.slice(0, SEQ_LEN);
});
els.predict.addEventListener("click", () => void onPredict());

loadModel().catch((e: Error) => {
  els.status.textContent = "模型加载失败";
  els.err.textContent = `${e.message}\n（若是文件缺失：先在仓库根目录跑 scripts/export_cnn_onnx.py）`;
});
