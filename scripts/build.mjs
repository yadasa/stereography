import { cp, mkdir, rm } from "node:fs/promises";

await rm("dist", { recursive: true, force: true });
await mkdir("dist", { recursive: true });
await Promise.all([
  cp("frontend/index.html", "dist/index.html"),
  cp("frontend/styles.css", "dist/styles.css"),
  cp("frontend/app.js", "dist/app.js"),
  cp("frontend/downloads", "dist/downloads", { recursive: true }),
  cp(".openai", "dist/.openai", { recursive: true }),
]);
