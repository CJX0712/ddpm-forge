# Diffusion Forge — DDPM 扩散模型实验室（forge 系列）

<p align="center">
  <a href="https://github.com/CJX0712/ddpm-forge/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/ddpm-forge/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/ddpm-forge/releases"><img src="https://img.shields.io/github/v/release/CJX0712/ddpm-forge?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/ddpm-forge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/ddpm-forge" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

**零依赖 · 单文件 HTML · 引擎可无头自检的 Denoising Diffusion 概率模型。**

从纯噪声到双月牙：可视化前向加噪 q(x_t|x_0) 与反向 ancestral 采样全过程，引擎 `DF.*` 无 DOM 依赖，可在 Node `vm` 中完整验证。浏览器打开 `index.html` 即用，无需构建、无需网络。

![license](https://img.shields.io/badge/license-MIT-green) ![deps](https://img.shields.io/badge/dependencies-0-brightgreen) [![verified](https://img.shields.io/badge/selftest-31%2F31%20%2B%2016%2F16-blue)](#可验证不变量)

## 快速开始

```bash
# 浏览器：直接打开 index.html → 选数据集 → 训练 → 生成演示

# 无头自检（Node ≥ 18，零依赖）：
node _smoke.js    # 31 项引擎不变量 → _smoke.log
node _probe.js    # ASCII 直方图/散点人工读数 → _probe.txt
node _uicheck.js  # DOM stub 冒烟，点遍全部控件 → _uicheck.log
```

## 引擎（`<script id="engine">`，全局 `DF`）

| 模块 | API | 说明 |
|---|---|---|
| 调度 | `makeSchedule(T,β₁,β_T)` | 线性 β ∈ (0,1)，ᾱ_t=Π(1−β_s)，严格单调递减 |
| 前向 | `qSample / qStats` | 闭式 q(x_t\|x₀)=N(√ᾱ·x₀,(1−ᾱ)I) |
| 网络 | `createNet / forwardNet / backwardNet / makeAdam` | 3 层 tanh MLP（hidden 16–64）+ 零初始化线性 skip + 手写 Adam |
| 时间 | `tEmbed` | [t, sin2πt, cos2πt, sin4πt, cos4πt] |
| 训练 | `train({data,dim,T,steps,batch,hidden,lr,seed})` | 去噪分数匹配：x_t=√ᾱ·x₀+√(1−ᾱ)·**ε**（同一 ε 配对！），预测 ε̂，MSE |
| 采样 | `sample / sampleOne` | ancestral：x_{t−1}=(x_t−β/√(1−ᾱ)·ε̂)/√α+√β·z，可记录轨迹动画 |
| 数据 | `makeMoons / datasetRing / datasetGauss2d / datasetGauss1d` | 零均值单位方差归一化 |
| RNG | `mulberry32 / gaussFactory` | Box-Muller，全程种子可复现（逐位确定性） |

## 可验证不变量（31 项 smoke 全绿）

| # | 不变量 | 实测 |
|---|---|---|
| 1 | 确定性：同 seed 训练/采样逐位一致 | lossCurve 全等 ✓ |
| 2 | β∈(0,1)，ᾱ 严格单调递减 | ✓ |
| 3 | 前向矩 = 闭式解（t=9/49/99，N=20000） | 偏差 < 6·SE ✓ |
| 4 | 高斯封闭性：x₀~N(0,1) ⇒ q(x_t)~N(0,1) | var=1±0.03 ✓ |
| 5 | 梯度检验：中心差分 vs 反向传播（含 skip） | maxRelErr < 1e-5 ✓ |
| 6 | **loss 地板 = E_t[ᾱ_t]**（最优去噪器残差） | 0.7494 vs 0.7418（+1.0%）✓ |
| 7 | **MSE 分解：loss ≈ 近似误差 + E_t[ᾱ_t]** | 0.7494−0.0054≈0.7418 ✓ |
| 8 | ε 条件均值近似误差 → 0 | 0.0054 ✓ |
| 9 | 生成矩（N(0,1) 目标，4000 样本） | mean −0.023，var 0.996 ✓ |
| 10 | 2D 双月牙训练收敛、生成有限在界 | loss 0.90→0.69 ✓ |

ASCII 探针（`_probe.txt`）：1D 生成直方图与 N(0,1) 逐 bin 吻合；2D 生成 700 点复现双月牙双弧结构。

## 调试实录（本项目最有价值的部分）

1. **ε 未配对 bug（真凶）**：最初实现里 `qSample` 内部自抽噪声 η 构造 x_t，而训练目标 ε 是另一次独立抽取 —— 目标与输入完全无关（E[ε|x_t]=0），网络"正确地"学到全零，loss 卡在 1.0，生成方差 4.4。二分定位（固定 t 能学 / 变 t 学不会 → 最终逐层排查数据构造）后改为**先抽 ε、用同一个 ε 构造 x_t**，一步修复：loss 0.98→0.749（地板 0.742），生成 var 3.9→0.996。
2. **loss 地板是 E_t[ᾱ_t] 而非 0**：ε-MSE 的不可约残差 = E_t[Var(ε|x_t)] = E_t[ᾱ_t]（单位方差数据）。"loss 降到 0"才是错的。断言写 `|loss/floor − 1| < 15%`。
3. **近似误差与地板要分开断言**：`E‖ε̂−ε‖²`（条件均值近似误差，→0）与训练 MSE（→地板）是两个不同的量；二者之差 ≈ E_t[ᾱ_t] 构成分解不变量。
4. **线性 skip（ResNet 风格）**：ε̂ 对 x_t 近似线性，零初始化直通路径让线性主信号直接传梯度，显著加速收敛。
5. **测试脚本自身的 rng 混用坑**：把高斯流当均匀 rng 抽 t → 负索引 → `abar[t]=undefined` → 静默 NaN。

## UI

- 数据集：双月牙 / 圆环 / 二维高斯 / 一维高斯（直方图模式，叠加真实密度曲线）
- 可调：T ∈ {50,100,200}、训练步数；loss 曲线叠加解析地板虚线；β/ᾱ 调度曲线
- 生成演示：300 条并行链从纯噪声逐帧退火至数据流形（附实时 ᾱ_t 读数）
- 全程中文、内联样式、零外部依赖、浅色主题

## forge 系列

nn / rl / tree / attn / pca / kmeans / gmm / hmm / gp / mcmc / kalman / svm / **diffusion** ——
统一范式：单文件 HTML、零依赖、可无头自检、每个算法至少一条交叉可验证的不变量。

## License

MIT © 晨星 (CJX0712)
