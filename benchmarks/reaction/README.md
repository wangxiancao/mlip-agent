# benchmarks/reaction/ — 反应能垒套件（Benchmark v3，PLAN §3.3）

对标 [Can Autonomous LLM Agents Execute Multireference Quantum Chemistry Calculations?](https://arxiv.org/html/2609.13357v1)
的 agent 能力评测：外部高精度金标准 + 六段复合链式任务 + 三档知识梯度 +
覆盖率/MAE/尝试次数/成本四维指标 + 失败归因讨论。

与 B 系列（`benchmarks/tasks.yaml`，框架级 canned 回归）分工：本套件面向
**live 模式评测 agent 的自主决策能力**，不替代 B 系列的回归门禁作用。

## 数据集

GMTKN55 **BH76** 子集（Goerigk et al., PCCP 2017, DOI 10.1039/C7CP04913G；
AIMNet2 官方论文的原生 benchmark，参考值为高级别 WFT）。提取 6 个反应
（`references/R{1..6}/`：物种 xyz + TS 初猜 + ref.json），类型覆盖单分子异构化、
氢转移、氢抽取、阴离子 S_N2、自由基加成、O(3P) 抽取——含自由基（mult=2/3）、
阴离子（charge=−1）与 Cl，套件模型为 **aimnet2**（元素/电荷/多重度支持匹配）。

重新生成：`python scripts/make_gmtkn_refs.py /path/to/GMTKN55`（数据源
`git clone https://github.com/grimme-lab/GMTKN55`）。

## 任务形态

每反应一条六段链：opt 反应物（们）→ opt 产物（们）→ freq 两极小（0 虚频）
→ 刚性扫描（沿断键）→ TS 搜索（prfo，初猜用数据集 ts.xyz）→ TS freq（恰 1 虚频）
→ 汇总正/反向能垒与反应能（kcal/mol）。

三档知识梯度（同一反应同一参考值，仅 prompt 信息量不同）：

- `R{n}-base`：只给结构文件、charge/mult 与一句话目标；
- `R{n}-dec`：base + `ladder.md` 决策阶梯全文（runner 自动注入）；
- `R{n}-repl`：完整工作流逐步写明（输入来源/参数/产物接力/换算公式）。

## 运行

```bash
# live 三档评测（需 MLIP_AGENT_LLM_*；全部 18 条）
.venv/bin/python benchmarks/runner.py --mode=live --suite=reaction
# 子集
.venv/bin/python benchmarks/runner.py --mode=live --suite=reaction --tasks R1-base,R1-dec,R1-repl

# canned 冒烟（R1-repl 剧本，真实 GPU 计算，验证管道）
.venv/bin/python benchmarks/runner.py --mode=canned --suite=reaction --tasks R1-repl

# 对已完成的运行重新判定
.venv/bin/python -m benchmarks.reaction.suite_judge benchmarks/reaction/runs/<ts>
```

产出：`runs/<ts>/suite_report.json`——coverage（得分参考量/总量）、
MAE（kcal/mol，mean/max，by_tier × by_reaction）、attempts（作业数 vs min_jobs、
一次成功率）、cost（GPU 秒 + 作业数）、failure_analysis（归因类别映射
ladder Rule A–I）与中文讨论段。

## 成本估算

每反应链 GPU 约 5–15 min（freq/TS 数值 Hessian 占大头）；
6 反应 × 3 档 ≈ 1.5–4 GPU 小时/轮 + LLM 时间。canned R1-repl 冒烟约 5–10 min。
