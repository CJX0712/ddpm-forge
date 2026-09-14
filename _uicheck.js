/* UI 冒烟：最小 DOM stub，加载 <script id="ui">，点遍所有控件 */
const fs = require("fs"), vm = require("vm"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const eng = html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1];
const ui = html.match(/<script id="ui">([\s\S]*?)<\/script>/)[1];

/* ---- 2d context 录制 stub ---- */
let drawCalls = 0;
function makeCtx2d(){
  return new Proxy({}, {
    get: function(t, k){
      if (k === "createImageData") return function(w, h){ return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; };
      if (k === "getImageData") return function(x, y, w, h){ return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; };
      if (k === "measureText") return function(){ return { width: 10 }; };
      if (typeof t[k] !== "undefined") return t[k];
      return function(){ drawCalls++; };
    },
    set: function(){ return true; }
  });
}

/* ---- 元素 stub ---- */
const elems = {};
function makeEl(id){
  return {
    id: id, value: "", innerHTML: "", textContent: "", style: {}, dataset: {},
    width: 400, height: 300, disabled: false, children: [],
    _h: {},
    addEventListener: function(type, fn){ this._h[type] = fn; },
    appendChild: function(c){ this.children.push(c); },
    getContext: function(){ if (!this._ctx) this._ctx = makeCtx2d(); return this._ctx; },
    focus: function(){},
    classList: { add: function(){}, remove: function(){}, toggle: function(){} }
  };
}

const pendingTimers = [];
const documentStub = {
  getElementById: function(id){ if (!elems[id]) elems[id] = makeEl(id); return elems[id]; },
  createElement: function(tag){ return makeEl("anon-" + tag); },
  addEventListener: function(){},
  body: makeEl("body")
};

const ctx = {
  console, Math, Error, Date, JSON, Object, Array, Float64Array, Int8Array,
  Uint8ClampedArray, isFinite, Infinity, NaN, parseInt, parseFloat,
  document: documentStub,
  requestAnimationFrame: function(cb){ cb(1e12); return 0; }, // 一次推到最后帧，终止动画
  setTimeout: function(fn, ms){ pendingTimers.push(fn); return pendingTimers.length; }
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(eng, ctx, { filename: "engine.js" });

/* select 默认值：真实浏览器里是第一个/带 selected 的 option；stub 必须显式模拟 */
(function seedSelects(){
  const re = /<select id="(\w+)">([\s\S]*?)<\/select>/g;
  let m2;
  while ((m2 = re.exec(html))){
    const opt = m2[2].match(/<option value="([^"]+)"(?:\s+selected)?/);
    if (opt) ctx.document.getElementById(m2[1]).value = opt[1];
  }
})();

let pass = 0; const fails = [];
function ok(name, cond, detail){
  if (cond) pass++; else fails.push(name + (detail ? " :: " + detail : ""));
}

vm.runInContext(ui, ctx, { filename: "ui.js" });
function flush(){
  let n = 0;
  while (pendingTimers.length && n++ < 20){
    const fns = pendingTimers.splice(0);
    for (const f of fns) f();
  }
}
const DFUI = ctx.DFUI;

/* 1. 初始化 */
ok("init-data-loaded", DFUI && DFUI.data && DFUI.data.kind === "2d", JSON.stringify(DFUI && DFUI.data && DFUI.data.kind));
ok("init-status", elems["status"] && elems["status"].textContent.length > 0);
ok("init-drew-canvas", drawCalls > 0);

/* 2. 一维高斯：训练 + 生成 */
elems["selData"].value = "gauss1d";
elems["selData"]._h.change();
ok("switch-1d", DFUI.data.kind === "1d");
elems["inpSteps"].value = "400";
elems["btnTrain"]._h.click();
flush();
ok("train-1d-model", !!DFUI.model && DFUI.model.lossCurve.length === 400);
ok("train-1d-status", /✅/.test(elems["status"].textContent), elems["status"].textContent);
elems["btnGen"]._h.click();
flush();
ok("gen-1d-samples", DFUI.samples && DFUI.samples.length === 600);
ok("gen-1d-finite", DFUI.samples && isFinite(DFUI.samples[0][0]));

/* 3. 双月牙：训练 + 生成动画 */
elems["selData"].value = "moons";
elems["selData"]._h.change();
elems["inpSteps"].value = "300";
elems["btnTrain"]._h.click();
flush();
ok("train-moons-model", !!DFUI.model);
elems["btnGen"]._h.click();
flush();
ok("gen-moons-frames", DFUI.frames && DFUI.frames.length > 3, String(DFUI.frames && DFUI.frames.length));
ok("gen-moons-samples", DFUI.samples && DFUI.samples.length === 600);
ok("gen-moons-finite", DFUI.samples && isFinite(DFUI.samples[0][0]) && Math.abs(DFUI.samples[0][0]) < 6);
ok("gen-status-ok", /✅/.test(elems["status"].textContent), elems["status"].textContent);

/* 4. 圆环快速过一遍 */
elems["selData"].value = "ring";
elems["selData"]._h.change();
elems["btnTrain"]._h.click();
flush();
elems["btnGen"]._h.click();
flush();
ok("ring-pipeline", !!DFUI.model && DFUI.samples && DFUI.samples.length === 600);

/* 5. 重置 */
elems["btnReset"]._h.click();
ok("reset-clears-model", DFUI.model === null && DFUI.samples === null);
ok("reset-status", elems["status"].textContent.length > 0);

/* 输出 */
const total = pass + fails.length;
const line = "PASS " + pass + " / " + total + "\n" + (fails.length ? "FAIL " + fails.join(" | ") : "ALL GREEN") + "\n";
fs.writeFileSync(path.join(__dirname, "_uicheck.log"), line);
console.log(line);
process.exit(fails.length ? 1 : 0);
