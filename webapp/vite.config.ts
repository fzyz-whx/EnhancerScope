import { defineConfig } from "vite";

// GitHub Pages 部署在仓库子路径下，用相对 base 保证资源路径正确
export default defineConfig({
  base: "./",
  build: { outDir: "dist", target: "es2022" },
});
