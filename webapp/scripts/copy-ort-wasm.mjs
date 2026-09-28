// 把 onnxruntime-web 的 WASM 运行时拷进 dist（避免运行时依赖 CDN，离线/弱网也能用）
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const src = join(root, "node_modules", "onnxruntime-web", "dist");
const dst = join(root, "dist");

if (!existsSync(src)) {
  console.warn("[copy-ort-wasm] 未找到 onnxruntime-web/dist，跳过（先 npm install）");
  process.exit(0);
}
mkdirSync(dst, { recursive: true });
let n = 0;
for (const f of readdirSync(src)) {
  if (f.endsWith(".wasm") || f.endsWith(".mjs") || f.endsWith(".js")) {
    cpSync(join(src, f), join(dst, f));
    n += 1;
  }
}
console.log(`[copy-ort-wasm] 复制 ${n} 个 WASM/JS 文件到 dist/`);
