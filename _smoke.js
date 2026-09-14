/* diffusion-forge 无头自检：所有不变量在 Node vm 中验证 */
const fs = require("fs"), vm = require("vm"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m){ console.error("engine script not found"); process.exit(1); }

const ctx = {
  console, Math, Error, JSON, Object, Array, Float64Array, Float32Array,
  Int8Array, Int32Array, Uint8Array, Uint8ClampedArray, isFinite, isNaN,
  parseInt, parseFloat, Infinity, NaN
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: "engine.js" });
const DF = ctx.DF;
if (!DF){ console.error("DF not exported"); process.exit(1); }

let pass = 0; const fails = [];
function ok(name, cond, detail){
  if (cond) pass++;
  else fails.push(name + (detail !== undefined ? " :: " + detail : ""));
}
function near(a, b, tol){ return Math.abs(a - b) <= tol; }

/* ---------- 1. 确定性：同 seed 训练 lossCurve 逐位一致 ---------- */
const data1 = DF.datasetGauss1d(512, 7);
const r1 = DF.train({ data: data1, dim: 1, T: 100, steps: 300, batch: 32, seed: 42 });
const r2 = DF.train({ data: data1, dim: 1, T: 100, steps: 300, batch: 32, seed: 42 });
let det = r1.lossCurve.length === r2.lossCurve.length;
for (let i = 0; i < r1.lossCurve.length && det; i++) det = r1.lossCurve[i] === r2.lossCurve[i];
ok("determinism-lossCurve-bitwise", det);
const s1 = DF.sample(r1, 5, 9), s2 = DF.sample(r1, 5, 9);
det = true;
for (let k = 0; k < 5 && det; k++) for (let i = 0; i < s1[k].length; i++) if (s1[k][i] !== s2[k][i]) det = false;
ok("determinism-sample-bitwise", det);

/* ---------- 2. 调度：β∈(0,1)，ᾱ 严格单调递减 ---------- */
const sched = DF.makeSchedule(100, 1e-4, 0.02);
let mono = true, inR = true;
for (let i = 0; i < 100; i++){
  if (sched.betas[i] <= 0 || sched.betas[i] >= 1) inR = false;
  if (i > 0 && !(sched.abar[i] < sched.abar[i - 1])) mono = false;
}
ok("beta-in-(0,1)", inR);
ok("abar-strictly-monotone", mono);
ok("abar-final-plausible", sched.abar[99] > 1e-4 && sched.abar[99] < 0.6, String(sched.abar[99]));
const sch1 = DF.makeSchedule(1, 1e-4, 0.02);
ok("schedule-T1-edge", sch1.betas[0] === 0.02 && near(sch1.abar[0], 0.98, 1e-15));

/* ---------- 3. 前向矩 = 闭式解 ---------- */
{
  const g = DF.gaussFactory(DF.mulberry32(99));
  const x0 = [0.7];
  for (const t of [9, 49, 99]){
    const N = 20000, st = DF.qStats(x0, t, sched);
    let sum = 0, sum2 = 0;
    for (let k = 0; k < N; k++){
      const xt = DF.qSample(x0, t, sched, g);
      sum += xt[0]; sum2 += xt[0] * xt[0];
    }
    const meanE = sum / N, varE = sum2 / N - meanE * meanE;
    ok("forward-mean-t" + t, near(meanE, st.mean[0], 6 * Math.sqrt(Math.max(st.var, 1e-6) / N)),
      "emp " + meanE.toFixed(5) + " vs " + st.mean[0].toFixed(5));
    ok("forward-var-t" + t, near(varE, st.var, 6 * st.var * Math.sqrt(2 / N) + 1e-6),
      "emp " + varE.toFixed(6) + " vs " + st.var.toFixed(6));
  }
}

/* ---------- 4. 高斯封闭性：x0~N(0,1) ⇒ q(x_t) 仍 N(0,1) ---------- */
{
  const g = DF.gaussFactory(DF.mulberry32(123));
  const N = 20000;
  let sum = 0, sum2 = 0;
  for (let k = 0; k < N; k++){
    const xt = DF.qSample([g()], 99, sched, g);
    sum += xt[0]; sum2 += xt[0] * xt[0];
  }
  const meanE = sum / N, varE = sum2 / N - meanE * meanE;
  ok("gauss-closure-mean", Math.abs(meanE) < 0.045, String(meanE.toFixed(4)));
  ok("gauss-closure-var", Math.abs(varE - 1) < 0.03, String(varE.toFixed(4)));
}

/* ---------- 5. 梯度检验（中心差分 vs 反向传播） ---------- */
{
  const net = DF.createNet(6, 5, 2, 123);
  const g5 = DF.gaussFactory(DF.mulberry32(5));
  const input = []; for (let i = 0; i < 6; i++) input.push(g5());
  const dOut = []; for (let i = 0; i < 2; i++) dOut.push(g5());
  const grads = DF.zeroGrads(net);
  const c = DF.forwardNet(net, input);
  DF.backwardNet(net, c, dOut, grads);
  function lossVal(){
    const cc = DF.forwardNet(net, input);
    let s = 0; for (let j = 0; j < 2; j++) s += dOut[j] * cc.out[j];
    return s;
  }
  const h = 1e-6;
  let maxRel = 0, worst = "";
  const groups = [["L1", net.L1, grads.L1], ["L2", net.L2, grads.L2], ["L3", net.L3, grads.L3], ["Ls", net.Ls, grads.Ls]];
  for (const [nm, L, gr] of groups){
    for (const key of ["W", "b"]){
      if (!L[key]) continue;
      const P = L[key], G = gr[key];
      for (let idx = 0; idx < P.length; idx++){
        const old = P[idx];
        P[idx] = old + h; const lp = lossVal();
        P[idx] = old - h; const lm = lossVal();
        P[idx] = old;
        const num = (lp - lm) / (2 * h), ana = G[idx];
        const rel = Math.abs(num - ana) / Math.max(1e-8, Math.abs(num) + Math.abs(ana));
        if (rel > maxRel){ maxRel = rel; worst = nm + "." + key + "[" + idx + "]"; }
      }
    }
  }
  ok("gradcheck-maxRelErr<1e-5", maxRel < 1e-5, maxRel.toExponential(3) + " @ " + worst);
}

/* ---------- 6. 训练 → loss 收敛到解析地板 E_t[ᾱ_t] ---------- */
const meanAbar = DF.meanAbar(sched);
const rMain = DF.train({ data: data1, dim: 1, T: 100, steps: 2000, batch: 64, seed: 42 });
{
  const L = rMain.lossCurve;
  let f = 0; for (let i = 0; i < 50; i++) f += L[i]; f /= 50;
  let l = 0; for (let i = L.length - 50; i < L.length; i++) l += L[i]; l /= 50;
  console.log("[diag] meanAbar=" + meanAbar.toFixed(4) + " first50=" + f.toFixed(4) + " last50=" + l.toFixed(4));
  ok("loss-floor-within-15pct", Math.abs(l / meanAbar - 1) < 0.15,
    "last50=" + l.toFixed(4) + " floor=" + meanAbar.toFixed(4));
  ok("loss-decreased", l < f, f.toFixed(4) + "->" + l.toFixed(4));
}

/* ---------- 7. ε 条件均值校验（独立估计路径）：E‖ε̂ − E[ε|x_t]‖² → 0 ---------- */
{
  const e = DF.epsTruthErr(rMain, 3000, 17);
  console.log("[diag] epsTruthErr=" + e.toFixed(4));
  ok("eps-condmean-approx<0.05", e < 0.05, "e=" + e.toFixed(4));
  // 分解不变量：MSE(=loss) ≈ 近似误差 + 方差地板 E_t[ᾱ_t]
  const L = rMain.lossCurve;
  let l = 0; for (let i = L.length - 50; i < L.length; i++) l += L[i]; l /= 50;
  ok("mse-decomposition", Math.abs((l - e) - meanAbar) < 0.05,
    "loss=" + l.toFixed(4) + " e=" + e.toFixed(4) + " floor=" + meanAbar.toFixed(4));
}

/* ---------- 8. 生成矩：N(0,1) 数据 → 生成样本应 N(0,1) ---------- */
{
  const smp = DF.sample(rMain, 4000, 7);
  const mo = DF.moments(smp);
  console.log("[diag] gen mean=" + mo.mean[0].toFixed(4) + " var=" + mo.var[0].toFixed(4));
  ok("gen-mean~0", Math.abs(mo.mean[0]) < 0.10, String(mo.mean[0].toFixed(4)));
  ok("gen-var~1", Math.abs(mo.var[0] - 1) < 0.12, String(mo.var[0].toFixed(4)));
  let finite = true;
  for (const s of smp) if (!isFinite(s[0])) finite = false;
  ok("gen-finite", finite);
}

/* ---------- 9. 边界 ---------- */
{
  const s1 = DF.sample(rMain, 1, 3);
  ok("sample-n1", s1.length === 1 && s1[0].length === 1 && isFinite(s1[0][0]));
  const rm = DF.train({ data: data1, dim: 1, T: 1, steps: 30, batch: 8, seed: 1 });
  ok("train-T1-runs", rm.lossCurve.length === 30 && isFinite(rm.lossCurve[29]));
  const qs = DF.qSample([1.0], 0, sched, DF.gaussFactory(DF.mulberry32(1)));
  ok("qSample-shape", qs.length === 1 && isFinite(qs[0]));
}

/* ---------- 10. 2D 双月牙：归一化 + 训练 + 生成有限 ---------- */
{
  const dm = DF.datasetMoons(600, 3, 0.06);
  const mo = DF.moments(dm);
  ok("moons-normalized-mean", Math.abs(mo.mean[0]) < 0.02 && Math.abs(mo.mean[1]) < 0.02,
    mo.mean[0].toFixed(4) + "," + mo.mean[1].toFixed(4));
  ok("moons-normalized-var", Math.abs(mo.var[0] - 1) < 0.05 && Math.abs(mo.var[1] - 1) < 0.05);
  const rm = DF.train({ data: dm, dim: 2, T: 100, steps: 1500, batch: 64, seed: 5 });
  const L = rm.lossCurve;
  let f = 0; for (let i = 0; i < 50; i++) f += L[i]; f /= 50;
  let l = 0; for (let i = L.length - 100; i < L.length; i++) l += L[i]; l /= 100;
  console.log("[diag] moons first50=" + f.toFixed(4) + " last100=" + l.toFixed(4) + " floor=" + meanAbar.toFixed(4));
  ok("moons-loss-decreased", l < f * 0.85, f.toFixed(3) + "->" + l.toFixed(3));
  ok("moons-loss-nan-free", isFinite(l));
  const smp = DF.sample(rm, 400, 11);
  let okBox = true;
  for (const s of smp) if (!(isFinite(s[0]) && isFinite(s[1]) && Math.abs(s[0]) < 5 && Math.abs(s[1]) < 5)) okBox = false;
  ok("moons-gen-in-box", okBox);
  // 圆环数据集归一化
  const dr = DF.datasetRing(500, 8, 0.08);
  const mr = DF.moments(dr);
  ok("ring-normalized", Math.abs(mr.mean[0]) < 0.02 && Math.abs(mr.var[1] - 1) < 0.05);
}

/* ---------- 输出 ---------- */
const total = pass + fails.length;
const line = "PASS " + pass + " / " + total + "\n" + (fails.length ? "FAIL " + fails.join(" | ") : "ALL GREEN") + "\n";
fs.writeFileSync(path.join(__dirname, "_smoke.log"), line);
console.log(line);
process.exit(fails.length ? 1 : 0);
