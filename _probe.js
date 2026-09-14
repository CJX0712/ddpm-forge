/* ASCII 探针：直方图 / 散点，供人工读数 —— asserts 全绿≠正确 */
const fs = require("fs"), vm = require("vm"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const eng = html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1];
const ctx = { console, Math, Error, Object, Array, Float64Array, Int8Array, isFinite, Infinity, NaN };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(eng, ctx, { filename: "engine.js" });
const DF = ctx.DF;

const out = [];
out.push("=== diffusion-forge probe ===");

/* --- 1D：生成样本直方图 vs N(0,1) --- */
const data1 = DF.datasetGauss1d(512, 7);
const r = DF.train({ data: data1, dim: 1, T: 100, steps: 2000, batch: 64, seed: 42 });
const smp = DF.sample(r, 4000, 7).map(function(s){ return s[0]; });
const B = 40, lo = -4, hi = 4;
function histOf(arr){
  const h = new Array(B).fill(0);
  for (const x of arr){ const i = Math.floor((x - lo) / (hi - lo) * B); if (i >= 0 && i < B) h[i]++; }
  return h;
}
const hg = histOf(smp), hd = histOf(DF.datasetGauss1d(4000, 555));
const mx = Math.max.apply(null, hg.concat(hd));
out.push("--- 生成样本(#, 4000) vs 目标 N(0,1) (=, 4000) ---");
for (let i = 0; i < B; i++){
  const x = lo + (hi - lo) * (i + 0.5) / B;
  const bar1 = "#".repeat(Math.round(hg[i] / mx * 40));
  const bar2 = "=".repeat(Math.round(hd[i] / mx * 40));
  const dens = Math.round(Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * 4000 * (B / 8) / mx * 40);
  out.push(x.toFixed(2).padStart(6) + " |" + bar1.padEnd(42) + "|" + bar2.padEnd(42) + "| true~" + dens);
}
const mo = DF.moments(DF.sample(r, 4000, 7));
out.push("gen moments: mean=" + mo.mean[0].toFixed(4) + " var=" + mo.var[0].toFixed(4) +
  "  |  loss floor E_t[abBar]=" + DF.meanAbar(r.sched).toFixed(4) +
  "  final loss=" + r.lossCurve[1999].toFixed(4) +
  "  epsTruthErr=" + DF.epsTruthErr(r, 2000, 17).toFixed(4));

/* --- 2D 双月牙：数据 '.' vs 生成 'o' --- */
const dm = DF.datasetMoons(600, 3, 0.06);
const rm = DF.train({ data: dm, dim: 2, T: 100, steps: 1500, batch: 64, seed: 5 });
const gen = DF.sample(rm, 700, 11);
const CW = 64, CH = 28, R = 3.4;
const canvasD = [], canvasG = [];
for (let y = 0; y < CH; y++){ canvasD.push(new Array(CW).fill(" ")); canvasG.push(new Array(CW).fill(" ")); }
function put(cv, x, y, ch){
  const cx = Math.floor((x + R) / (2 * R) * (CW - 1));
  const cy = Math.floor((R - y) / (2 * R) * (CH - 1));
  if (cx >= 0 && cx < CW && cy >= 0 && cy < CH) cv[cy][cx] = ch;
}
for (const p of dm) put(canvasD, p[0], p[1], ".");
for (const p of gen) put(canvasG, p[0], p[1], "o");
out.push("--- 双月牙 生成(o,700) vs 数据(.,600)（合并视图：o 盖 .）---");
for (let y = 0; y < CH; y++){
  let line = "";
  for (let x = 0; x < CW; x++){
    const g = canvasG[y][x] === "o", d = canvasD[y][x] === ".";
    line += g ? "o" : (d ? "." : " ");
  }
  out.push(line);
}
const mg = DF.moments(gen);
out.push("gen moments: mean=(" + mg.mean[0].toFixed(3) + "," + mg.mean[1].toFixed(3) +
  ") var=(" + mg.var[0].toFixed(3) + "," + mg.var[1].toFixed(3) + ")");
out.push("moons final loss=" + rm.lossCurve[1499].toFixed(4));

/* --- 采样轨迹快照（单链） --- */
const g9 = DF.gaussFactory(DF.mulberry32(9));
const one = DF.sampleOne(rm, g9, 25);
out.push("--- 单链轨迹 (t: 99→0, 每 25 步) ---");
for (const fr of one.traj)
  out.push("t=" + String(fr.t).padStart(3) + "  x=(" + fr.x[0].toFixed(3) + ", " + fr.x[1].toFixed(3) + ")");

fs.writeFileSync(path.join(__dirname, "_probe.txt"), out.join("\n") + "\n");
console.log("probe written");
