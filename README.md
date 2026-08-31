# fence_lite —— 图纸上的 fence 文字、符号、引线与线型

给几个月后接手的人：这是当前生产服务 <http://18.222.147.166:5051/> 的项目说明。
上传一份施工图 PDF，把每一页里跟 fence 有关的**文字、符号、callout 引线和末端实际
指到的线型**标出来。

## 1. 这个服务干什么

1. 用户上传 PDF；
2. 每一页找出 fence 相关**文字**，在图上标注最终位置；
3. 文字如果落在 legend / schedule / note_cluster 这类图例表格里，就看它左边有没有
   配套的**样例图形**，分成两类：
   - `shape` —— 闭合外框里带短码的符号（比如一个六边形里写 `SF-1`）
   - `line`  —— 线型样例（比如一段带叉的虚线）
4. `shape` 样例 → 在**俯视图（plan）视图**里把全页同款符号都匹配出来（本地矢量几何，
   零模型成本）；
5. callout / 放置锚 → 本地箭头边车恢复引线与末端；
6. 独立 Python 边车识别整页矢量线型：callout 末端绑定它实际指到的线型；无可靠
   引线的同款 symbol 用框中心的局部候选形成共识；legend 的 `line` 样例则作为
   supervised template 直接提取并匹配全图（零模型成本）。

最终用户看到的：**所有 fence 文字位置 + fence 相关 symbol 位置（样例框 + plan 视图内的
全部放置）+ callout 引线 / 箭头 + 末端实际指到的线型高亮**。

### 当前生产代码基线

本轮整理前，本地当前算法源码已与 `18.222.147.166:5051` 远端磁盘版本逐字节校验
一致；相关源码的修改时间也早于生产 worker 启动时间。本文因此以该 worker 实际
加载、且生产开关当前会走到的调用链为准，不再把历史 Windows 副本当作实现基线。

砍掉的（不要再找了，代码里一行都没有）：

| 砍掉的东西 | 为什么 |
|---|---|
| 旧参考项目的 fenceline 全套（三票定位、联合视觉裁判、representative 同文择优、`callout_selections.json`、`fencelines.json`、Notion 测试集、`/api/rescan`） | 不再沿用旧的整条围栏线流水线 |
| 旧 `line` 整线矢量追踪（`find_fence_vectors` / `char_filter` / trace / 全部 line descriptor 与 sweep 机制） | 已由步骤6的独立线型引擎聚类 + callout 末端绑定替代 |
| `fence_text_scan/` 历史 base 存储 + `primary_rescue` 三级优先 | 新项目只有一个 `vlm.json`，一处真相 |
| 交互式付费接口 | 所有付费调用只发生在上传作业里，网页端**只读缓存** |

## 2. 一页 PDF 的六阶段流程

```
                        projects/<slug>/input.pdf
                                  │
   ┌──────────────────────────────┴──────────────────────────────┐
   │ 步骤1  找 fence 文字（唯一「零遗漏」保证的一步）             │  💰 付费
   │                                                              │
   │  输入：整页渲染图（144 DPI，长边 ≤5000）+ PDF 原生文字层      │
   │  算法：a) 矢量层抽排版行 → 文字判词（LLM 判「这句是 fence 吗」，│
   │           带关键词兜底）→ fence 实例                          │
   │        b) 整页图丢给 Gemini，直接返回带框的文字记录            │
   │        c) 融合：VLM 框先吸附到真实文字上，再和矢量实例取并集   │
   │           —— 判词认了但 VLM 没框到的，一定作为 vec_added 补进来 │
   │  输出：data/<slug>/results.json                               │
   │        rec = {vlm_items, vec_added, vec_covered, has_text, …} │
   │        items_of(rec) = vlm_items + vec_added ← 全栈 union index │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
   ┌──────────────────────────────┴──────────────────────────────┐
   │ 步骤2  找图例样例符号                                        │  💰 付费
   │                                                              │
   │  输入：整页渲染图 + 步骤1 的文字框列表（带 index）             │
   │  算法：一次推理同时给出「页面分组」和「样例符号」，再过两条     │
   │        确定性硬闸：① 每个 symbol 必须有合法 text_index 主人；   │
   │        ② 必须落在 legend/schedule/note_cluster 组里            │
   │        （view 区的 plan 泡泡、详图标题圈、指北针一律剥掉）      │
   │  输出：data/<slug>/symbols.json                               │
   │        {sig, v, pv, model, raw, result:{symbols, groups}}     │
   │        raw = 未过滤的付费原文，留着让将来的过滤版本免费重跑     │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
   ┌──────────────────────────────┴──────────────────────────────┐
   │ 步骤3  视图投影分类                                          │  💰 付费（很便宜）
   │                                                              │
   │  输入：整页渲染图 + 步骤2 里 kind=view 的组框                 │
   │  算法：让模型判每个 view 是 plan / elevation / section /       │
   │        detail / other。**故意是一份独立缓存**：分类逻辑改了     │
   │        只重付分类，不动步骤2 的 symbol raw                     │
   │  输出：data/<slug>/view_types.json  {sig, v, model, views}    │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
   ┌──────────────────────────────┴──────────────────────────────┐
   │ 步骤4  shape 符号全页放置匹配                                │  🆓 免费
   │                                                              │
   │  输入：PDF 矢量几何 + shape 样例框 + 步骤3 认定的 plan 组框     │
   │  算法：纯本地几何 —— 拿样例框里的闭合形状当模板，在全页矢量里   │
   │        找同款，再用 plan 组框过滤（**fail-closed**：这一页没有  │
   │        plan 组就一个放置都不留）。line 类留给步骤6处理          │
   │  输出：symbols.json 的 result 上加 placements + plc_v          │
   │        + plc_scope_sig（模板与 plan 框共同决定缓存身份）       │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
   ┌──────────────────────────────┴──────────────────────────────┐
   │ 步骤5  箭头 / 引线恢复                                      │  🆓 免费
   │                                                              │
   │  输入：PDF 矢量几何 + 文字 / shape 放置锚                     │
   │  算法：Node 一次性边车恢复引线拓扑和箭头末端，几何兜底补回     │
   │        边车漏掉的编号标记                                    │
   │  输出：data/<slug>/arrows.json                               │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
   ┌──────────────────────────────┴──────────────────────────────┐
   │ 步骤6  线型聚类、末端绑定与高亮                              │  🆓 免费
   │                                                              │
   │  输入：PDF 矢量几何 + callout 末端 / symbol 中心 / legend 样例 │
   │  算法：source-aligned 引擎的 Method 1 + Method 2 聚类；剔除    │
   │        marker 自身几何，末端按最近 path op 绑定；同款无引线    │
   │        symbol 以共同候选投票；legend 水平样例走监督模板匹配     │
   │  输出：data/<slug>/linetypes/<page>.json                        │
   │        data/<slug>/legend_linetypes/<page>.json                │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                        webapp 只读这些 JSON 画层
```

费用只发生在步骤 1/2/3，全部经过 `core.gemini.gen_json` 这**一个**出口，
`core.gemini.RECORDER` 顺手记下调用数、模型秒数、token、USD、峰值并发。

## 3. 目录结构

```
fence_lite/
├─ run.ps1                借 fence_detector 的 venv 起 webapp（端口 5060）
├─ .env                   GEMINI_API_KEY=...（不进 git）
├─ webapp.py              Flask：上传 / 画廊 / 单页 JSON / 底图，只读缓存
├─ job.py                 上传作业编排：默认 fence 目标六阶段依次跑，唯一会花钱的地方
├─ core/
│  ├─ config.py           env / 模型价目表 / DPI 与像素上限 / compute_cost
│  ├─ gemini.py           gen_json（全项目唯一 generate_content 出口）+ RECORDER
│  ├─ pdfio.py            FITZ_LOCK（进程级锁）+ render_pdf_page
│  └─ parsing.py          JSON / box 解析与归一化小工具
├─ steps/
│  ├─ store.py            data/<slug>/ 磁盘布局与读写（磁盘契约都写在它的 docstring 里）
│  ├─ versions.py         全部缓存失效开关（见下面第 6 节）
│  ├─ debug.py            DebugSink：默认 None 时所有记录点零开销
│  ├─ text/               步骤1
│  │  ├─ target.py        唯一可编辑的检测目标 + 两个固定任务外壳（图 / 判词）
│  │  ├─ vector_layer.py  PDF 原生文字层 → 0-1000 框（多进程，MuPDF 非线程安全）
│  │  ├─ judge.py         文字判词缓存 + 关键词兜底 + 分块并发
│  │  ├─ vlm_layer.py     整页图扫描（含扫描页双模型并集）
│  │  ├─ vlm_cache.py     raw 记录的身份契约（pdf_revision + model + prompt 摘要）
│  │  ├─ cleanup.py       VLM 框校验、吸附到真实文字、剥离标记码
│  │  ├─ merge.py         融合（零遗漏保证）
│  │  └─ debug_view.py    免费的调试视图
│  ├─ prompts.py          步骤2 的提示词 + response schema + 分组类别
│  ├─ symbols.py          步骤2：一次推理 + 两条硬闸 + 缓存编排
│  ├─ views.py            步骤3：视图投影分类（独立付费缓存）
│  ├─ placements.py       步骤4：本地矢量 shape 匹配 + plan 过滤（免费）
│  ├─ arrows.py           步骤5：箭头 / 引线 Node 边车 + 取景 + 回映射（免费）
│  ├─ linetypes/          步骤6：线型边车调用、末端绑定、显示闸与逐页缓存（免费）
│  └─ leaders.py          放置锚引线的几何兜底（边车漏掉的编号标记，见第 8 节）
├─ tools/
│  ├─ import_project.py   把 5051 已算好的项目搬进来，零花费（见第 5 节）
│  ├─ check_rerun_cost.py 重跑前先算钱：只读盘，比对版本戳 + 模型（见第 6 节）
│  └─ compare_runs.py     两次运行的量化对比（同一 PDF 两个模型，见第 7b 节）
├─ templates/             前端页面
├─ projects/<slug>/input.pdf   源 PDF
├─ data/<slug>/*.json          全部缓存（布局见 steps/store.py）
└─ _jobs/                      上传作业的状态与日志
```

## 4. 跑起来

```powershell
./run.ps1        # → http://127.0.0.1:5060/
```

- 端口 **5060**（5051 是服务器上的生产，5055 是本机 fence_takeoff_web，别撞）。
- 从 `.env.example` 复制 `.env` 并填写 API key。`ARROWS=1` 与 `LINETYPES=1`
  是完整客户流程的硬要求；标准入口会在监听端口和付费处理前验证两项开关、Node
  边车、普通/legend/调试线型边车及隔离 Python 依赖，缺任何一项都拒绝启动，
  不再把四阶段结果显示为完整 Done。
- 依赖**借用** `C:\Users\Administrator\fence_detector\venv`（同一套 flask + PyMuPDF + google-genai +
  Pillow，不要再建一个 venv）。`run.ps1` 里可用 `FENCE_PYTHON` 覆盖解释器。
- 手工跑脚本 / 测试：
  ```powershell
  $env:PYTHONUTF8=1
  C:\Users\Administrator\fence_detector\venv\Scripts\python.exe -B -m unittest discover -s tests
  ```
  cwd 必须是项目根（import 根是 `fence_lite`，风格 `from core.x import y` / `from steps.x import y`）。
- **改了后端必须先杀掉占着 5060 的旧进程再起**，否则你看到的是老代码。

## 5. 导入已有项目（零花费）

```powershell
python -B tools/import_project.py                       # 列出可导入的 slug
python -B tools/import_project.py koch_tennis_center --dry-run
python -B tools/import_project.py koch_tennis_center rapid_city
python -B tools/import_project.py my_slug --from D:\some\other\fence_takeoff_web
```

从参考项目 `C:\Users\Administrator\fence_takeoff_web` 把已经算好的项目搬过来，
立刻能在新界面里看结果，**不花一分钱**。文件名映射：

| 源 `<REF>/fence_fused/<slug>/` | 目标 `data/<slug>/` |
|---|---|
| `vlm_extra.json` | `vlm.json`（改名） |
| `results.json` / `vec.json` / `textjudge.json` / `symbols.json` / `view_types.json` / `vlm_flash.json` | 同名 |
| `callout_selections.json` / `fencelines.json` | **不搬**（已砍的旧 fenceline 流水线） |
| `base_P*.jpg` | **不搬**（底图按新 revision 自动重生） |

### 坑：跨机搬项目 pdf_revision 会变

所有付费缓存的主键里都混了

```
pdf_revision = f"{size:x}-{mtime_ns:x}"      # steps.store.pdf_revision
```

文件一拷贝 mtime 就变（ext4 → NTFS 还会把纳秒 round 到 100ns），于是**内容完全相同**
的 PDF 得到一个新 revision，所有 sig 全部失配 —— 管线会把已经付过钱的推理再买一遍。
所以导入工具的核心工作不是拷文件，而是**重算签名**（一律调用本项目自己的函数，
公式只有一处、天然正确）：

- `results.json` 顶层 `pdf_revision` + 每页 `vlm_sources[].identity.pdf_revision`
- `vlm.json` / `vlm_flash.json` 每条 `vlm_identity.pdf_revision`
- `symbols.json` 每页 `sig = sig_of(items_of(rec), 新revision)`
- `view_types.json` 每页 `sig = view_signature(raw_groups, 新revision, model)`
- `vec.json` 的 `pdf_mtime`（这个是 `st_mtime` 浮点，不是 revision；不改的话矢量层会
  全量重抽 —— 免费但慢）

导入结束会打印一份自检表（页数 / 文字项 / symbol 数 / placements 数 / 签名是否匹配）。
两点必看：

- **旧的 `placements` 会被剥离，然后就地重算**。旧的是旧语义（没有 plan 视图过滤，而且
  line 类也跑过整线传播），所以导入时连 `trace / line_type / sample_evidence /
  vec_error / prop_v / debug` 一起剥掉，只留付费的 `raw` 和 `result.{symbols,groups}`；
  紧接着调 `steps.placements.match_placements` 用**新语义**重算一遍（本地矢量几何，
  零 Gemini 花费）。所以自检表里的 placements 数**和生产 5051 的数字不会相等** ——
  实测 koch_tennis_center 1239 → 1225、rapid_city 4 → 3（其中 1 个被 plan 过滤掉）。
  哪一页的视图分类不当期（sig / model / `VIEW_VERSION` 对不上），这一页就 fail-closed
  跳过、留给管线自己跑，日志里会写「N 页等视图分类」。
- 已存在同名项目会**直接拒绝**（不静默覆盖）。要重来先删 `projects/<slug>` 和 `data/<slug>`。

另外注意：生产的 `vlm_extra.json` 只覆盖**部分**页（其余页的 raw 在已砍掉的
`fence_text_scan/` 历史 base 存储里）。`results.json` 是完整的，所以**看**没问题；
但如果之后 bump 了步骤1 的付费缓存键，缺 raw 的那些页会重新付费。

## 6. 缓存与版本旋钮（`steps/versions.py`）

数值故意与生产 5051 保持一致，这样导进来的旧缓存仍然算当期。

| 常量 | 控制什么 | bump 的代价 |
|---|---|---|
| `FUSED_VERSION = 2` | `results.json` 的融合语义（vlm_items / vec_added / vec_covered 三桶口径） | 🆓 免费 —— 从已缓存的 vec / 判词 / VLM raw 重建 |
| `TEXT_JUDGE_VERSION = 2` | `textjudge.json` 的判词语义（无 `v` 的旧缓存视为 1） | 💰 重判所有字符串（纯文本，最便宜的一档） |
| `SYMBOL_PROMPT_V = 17` | 步骤2 的提示词 / response schema。**改 `steps/prompts.py` 必须同步 bump** | 💰 每页重新推理 |
| `SYMBOL_VERSION = 19` | 步骤2 的发布过滤 schema（owner 硬闸 + 组内硬闸） | 🆓 只要 `SYMBOL_PROMPT_V` 不变，就是拿已存 raw 免费重过滤 |
| `VIEW_VERSION = 1` | 步骤3 分类器（taxonomy / 提示词 / 判读策略） | 💰 只重付分类，不动 symbol raw |
| `PLACEMENT_VERSION = 4` | 步骤4 本地 shape 匹配 + plan 过滤；包含分节行号继承与输入 scope 签名 | 🆓 纯几何，零模型调用 |
| `steps/arrows.py: ARROWS_VERSION = 17` | 步骤5箭头语义（边车 + `steps/leaders.py` 的几何兜底） | 🆓 本地边车 + 纯几何，零模型调用 |
| `steps/linetypes/version.py: LINETYPE_VERSION = 6` | 步骤6绑定 / 发布语义；支持无引线 symbol 中心共识，签名另含锚点、边车源码与依赖摘要 | 🆓 本地边车 + 纯几何；所有项目逐页重算 |
| `steps/legend_linetypes: VERSION = 1` | legend 线样例监督匹配通道 | 🆓 本地边车 + 纯几何；只重算含 line 样例的页 |

不在这张表里但同样会作废缓存的东西：

- **检测目标文本**（`steps/text/target.py` 的 `TARGET_DEFAULT`，用户在上传对话框里可改）
  —— 它被 sha256 进 `vlm_identity.prompt_sha256`，改一个字就是全量重付步骤1。
- **模型 id**（`GEMINI_MODEL`，或对比运行由 slug 钉定的模型）—— 进 `vlm_identity`、
  `view_types` 的 sig，以及 `symbols` 的 `model` 字段。
- **PDF 本身**（见第 5 节的 revision）。

### ⚠️ bump 是全局的，不是只影响你手上那个项目

版本戳进的是每个项目自己的 sig，所以 **bump 一个常量会让盘上所有项目的对应
缓存同时失效**。失效的缓存不会被删，但**发布闸不放行** —— 界面上直接变成
「这一层没有结果」。实测踩过：把 `ARROWS_VERSION` 9→10 之后只重跑了一个项目，
另外 10 个项目的引线（paducah 167 条、taylor 161 条、rapid_city 47 条…）全部
停止显示，直到逐个重跑才回来。**bump 之后要把所有项目都重跑一遍。**

### 重跑之前先算钱

```sh
venv/bin/python tools/check_rerun_cost.py <slug>
```

只读盘、零调用，逐个付费步比对版本戳**和模型**，告诉你这次重跑会不会花钱。
两个它专门防的坑：

- **没跑完的项目**：整项目 rerun 会把没跑完的页一起补上 —— 那是真实的新增付费。
  （实测：一个 73 页的 PDF 只跑完 9 页，一次"以为免费"的重跑花了 $1.15。）
- **对比运行**：变体的缓存是按它自己的模型盖戳的。重跑时若没把模型钉回去，
  所有按 `resolve_model(None)` 校验的缓存都读作过期，会用默认模型重新付费
  **并覆盖掉那份对比结果**。`/api/rerun` 现在会自动从 slug 把模型钉回去。

默认准确率模式下，`vlm.json` 和 `vlm_flash.json` 都应有每一页的当期记录。
页上即使有可提取的标题栏文字，真正的 CAD 标注仍可能是纯描边 path，
所以不能据此跳过读图。只有显式设 `SCAN_ALL_PAGES=0` 才回到选择性省钱口径。

## 7. 并发 / 成本旋钮

都是环境变量，`job.py` 会 `setdefault` 一份默认值。

| 变量 | 默认 | 干什么 | 调高的风险 |
|---|---|---|---|
| `MAX_PARALLEL_JOBS` | 按 CPU/内存推导（本机 2） | 同时处理多少份 PDF；其余保持 queued | 多份大图同时渲染/聚类会抬高内存峰值 |
| `GEMINI_MAX_CONCURRENCY` | 8 | 所有 PDF、Web/维护进程合计的 Gemini 在途调用上限 | 429 / 配额；必须让所有进程使用同一个值 |
| `HEAVY_SIDECAR_SLOTS` | 按页级 worker 推导（生产显式 3） | 箭头与线型边车共享的跨进程总槽位 | CPU/内存超订；Web 与 refresh 必须使用同一个值 |
| `TEXT_WORKERS` | 8 | 步骤1 整页 VLM 扫描，页级并行（线程） | 429 / 配额；每个 worker 都要先渲染一页，而渲染在 `FITZ_LOCK` 里**串行** |
| `SCAN_ALL_PAGES` | 1 | 每页 Pro + Flash 取并集，避免混合 CAD 页假阴性 | 步骤1调用量增加；设 0 会重新漏掉纯 path 字样 |
| `VEC_WORKERS` | `min(cpu, 6)` | 矢量文字层抽取，页级并行（**进程**，MuPDF 非线程安全） | 内存（大图纸每页几十 MB）；页数 < `VEC_MIN_PARALLEL`(24) 时自动退回单进程 |
| `JUDGE_WORKERS` | 4 | 文字判词分块并行（纯网络） | 429；判词是最便宜的一档，没必要拉太高 |
| `SYMBOLS_WORKERS` | 8 | 步骤2 页级并行 | 429 + `FITZ_LOCK` 渲染串行 |
| `VIEW_WORKERS` | 6 | 步骤3 页级并行 | 同上 |
| `LINETYPE_PAGE_WORKERS` | 按 CPU 推导（生产显式 3） | 步骤6同时运行多少个单页边车 | 与页内 CPU budget 相乘后超订，且每页几何内存峰值不小 |
| `LINETYPE_CPU_BUDGET` | 按 CPU 推导（生产前台 4；refresh 2） | 单页边车内部引擎的 worker 预算 | 高于甜点位通常不再提速，还会和其他页抢核 |
| `LINETYPE_TIMEOUT` | 600 | 普通线型页硬上限（秒） | 过低会丢密页线型 |
| `LINETYPE_DENSE_TIMEOUT` | 3600 | >=40k vector paths 密页上限 | 过高会让异常密页长时间占队列 |

当前生产前台是 **3 页并发 × 每页 budget 4**；低优先级
`fence-linetype-refresh.service` 是 **1 页并发 × 每页 budget 2**。两者共享
`HEAVY_SIDECAR_SLOTS=3`，refresh 发现前台上传 / 重跑时不会再提交新页。

真正的瓶颈通常**不是** Gemini 而是渲染：`core.pdfio.FITZ_LOCK` 是进程级 RLock，
所有 `fitz` 调用点都必须持它（MuPDF 在同一进程里不是线程安全的，就算各自开
`Document` 也会因为共享的分配器 / 字体状态段错误）。所以把 workers 从 8 拉到 24
往往只是让更多线程排队等锁。要提渲染吞吐得走**进程**（矢量层就是这么干的）。

费用与耗时看 `core.gemini.RECORDER.summary()`：`calls / model_seconds /
input_tokens / output_tokens / thoughts_tokens / cost_usd / peak_concurrency / by_model`。
`model_seconds` 是**模型忙时总和**（并发会叠加），不是墙上时间 —— 两个都要报才看得懂。
`peak_concurrency` 是实测的真实并行度，用来判断 workers 有没有真的生效。

## 7b. 换模型 / 提供方对比（Gemini ⇄ Claude）

同一份 PDF 可以用另一个模型再跑一遍，**原来的结果一个字节都不动**，两份并排放着对比。

### 怎么用

**界面上一个 PDF 只有一行。** 行内每个跑过的模型一个 chip，实心的是当前显示的：

```
gladstone_dog_park                          ⇄  ↻  ✕
[●Gemini 3.1 Pro] [○Claude Sonnet 5] [fence] [8 页] [文字 42] [$0.3344]
```

- 点 **chip** 切模型：换的是这一行显示的整份结果（文字/符号/放置/费用一起换）
- 行标题的 **`⇄`** 发起一次新的对比运行（选模型）；**`↻`** 重跑当前选中那一份；
  **`✕`** 删整个 PDF（**级联**删掉它的所有对比运行）
- 打开某一页之后，**顶栏**还有一个 `模型 | [Gemini] [Claude]` 开关，
  切换时**保持页码不变** —— 对比要看同一张图才有意义

磁盘上仍然是两个目录：`data/<slug>/` 和 `data/<slug>__<model>/`。**这个必须分开**：
缓存当期判定带模型（`steps/symbols.py` 的 `entry["model"] == resolve_model(None)`），
混在一个目录里会让两边互相作废、还会让一次对比里混进另一个提供方的付费 raw。
`__<model>` 后缀是存储实现，界面上不出现（`baseSlug()` / `slugModel()` 负责剥掉）。

命令行等价物：

```sh
curl -X POST http://127.0.0.1:5051/api/variant/<slug> \
  -H 'Content-Type: application/json' -d '{"model":"claude-sonnet-5"}'
```

量化对比（只读盘，零花费）：

```sh
venv/bin/python tools/compare_runs.py <slug-A> <slug-B>
```

它按**文字归一化后相同**的条目配对，报每页的召回差、以及配对上的条目的框
IoU / 中心误差。只比数量会漏掉最要紧的一类失败：两边找到同一句话、但框画在
不同地方 —— 那样叠加层是错的，而计数看起来一模一样。

### 实现在哪

- `core/config.py` —— `PRICING` 里加了 Claude 条目（带 `provider` 字段）；
  `set_model_override()` / `resolve_model()` 提供**作业级**模型钉定。作用域和
  `RECORDER` 一样：在 `_PROC_LOCK` 里设、`finally` 里清，靠「作业串行 + `-w 1`」
  成立。显式传入的 `model=` 参数优先级更高。
- `core/llm.py` —— Anthropic 后端。返回对象刻意伪装成 google-genai 的响应
  （`.text` + Gemini 字段名的 `.usage_metadata`），所以 `usage_from_response`、
  `RECORDER`、`compute_cost` 全都不用改。
- `core/gemini.py::gen_json` —— 仍然是全项目唯一付费入口，只是按 model id 分派。

**提示词不按提供方改写**：`steps/prompts.py` 逐字发给两边，否则对比就没意义了。
Claude 需要的额外交代（只输出裸 JSON、box_2d 是 0-1000 归一化而不是像素）放在
system prompt 里 —— 那是 adapter 的职责，不是任务提示词的一部分。

### Claude 侧的旋钮

| 变量 | 默认 | 干什么 |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | 必填，放 `.env` |
| `ANTHROPIC_MAX_CONCURRENCY` | `1` | 并发上限。**实测**当前 key 只允许 1 个并发连接（2/3/4/6/8 并发各试过，每次只有 1 个成功，其余 429）。限额提上去之后调大 |
| `ANTHROPIC_EFFORT` | `high` | 推理深度（`low`/`medium`/`high`/`xhigh`/`max`） |
| `ANTHROPIC_MAX_TOKENS` | `16000` | thinking + 可见输出的合计上限 |
| `ANTHROPIC_COORD_MODE` | `normalized` | `pixel` 改成要绝对像素坐标、由 adapter 换回 0-1000。官方文档推荐 pixel，但**本工况实测更差**（gladstone P2：平均 IoU 0.189 vs 0.287），所以默认留 normalized |

### 三个已经踩过的坑

1. **并发连接 429**。流水线按 6 线程扇出，Gemini 实测峰值 5 并发都没事；Anthropic
   账号另有一条 *concurrent connections* 限额，越限的 429 靠重试解决不了（每个
   worker 的重试会继续相撞）。闸设在 `core/llm.py` 这一侧，不动共享的 `*_WORKERS`，
   免得把 Gemini 的并行度一起拖下来。
2. **图必须自己先缩放**。Claude 返回的坐标是**服务端缩放之后**那张图的坐标系。
   4896x3168 的图纸是 19950 visual tokens、远超 4784 上限，服务端会静默降采样，
   于是模型量的是一个本地代码从没算过的帧 —— 实测表现是 y 值跑到 1163~1256、
   越过 1000 被 `is_normalized_box` 拒掉。`_target_size()` 按官方算法先缩到
   2380x1540 再发，坐标就对上了。注意**是 token 预算在起作用、不是边长上限**：
   按 2576 边长去缩会每个坐标都偏。
3. **不要 clamp 越界的框**。`core/parsing.py` 的注释已经写了原因：clamp 出来的框
   会落在页边、看着像真的，几何上是错的。越界就让它失败。

### 已知差异（gladstone 实测）

Sonnet 5 在**结构化列表里的文字**上和 Gemini 3.1 Pro 框得几乎一样（IoU ~0.70），
但**平面图里游离的 callout** 会偏 5~8% 页宽（IoU 0.00）。文字本身找得到，位置不准。
这是工况上的能力差异，不是 adapter 的 bug —— 两种坐标模式都试过了。

## 8. 箭头与线型模块（步骤5 / 6）

生产已启用 `ARROWS=1` 和 `LINETYPES=1`。`job.py` 在 placements 之后先按页运行
`steps/arrows.py`，写 `data/<slug>/arrows.json`；线型层随后按页启动
`tools/linetype_sidecar/run.py`，聚类、绑定后原子写入
`data/<slug>/linetypes/<page>.json`。两步都是本地计算，不调用模型。

线型边车的进程协议是**一进程一页**：stdin 一个 JSON，stdout 一个 JSON；父层用
`LINETYPE_PAGE_WORKERS` 并发多个页面，并在每页完成后独立校验先决条件与落盘。

**必须遵守的三条契约**（违反就是静默错位，不会报错）：

1. **union index 顺序**。`items = steps.store.items_of(rec)`，拼接顺序是
   `vlm_items` 在前、`vec_added` 紧随其后，**下标就是全栈公共编号** ——
   symbol 的 `text_index`、前端选中态的编号、箭头返回 dict 的键，全锚在这一个下标上。
   调整拼接顺序或往中间插项 = 静默错位所有归属。
2. **坐标帧**。所有框都是页面帧 `[ymin, xmin, ymax, xmax]`、点是 `[y, x]`，
   0-1000 闭区间，和文字框 / symbol 框完全同帧。模型只在它看到的那张图里作答，
   **回映射到页面帧是调用方的责任**（裁了图就得自己映回去，见
   `steps/debug.py: px_box_to_page`）。
3. **返回值只含真的找到箭头的锚**，键是 union index；线型层只会处理这些结果里的
   有效末端，并先剔除该 callout 自身的引线 / 箭头笔画。

箭头和线型的缓存都 fail-closed：签名不当期或计算失败时不发布旧几何；线型失败页会
保留显式错误并在后续运行或低优先级 refresh 中重试。

## 9. 已知边界

- **整页视觉仍不是数学上的零漏检**。默认每页都由 Pro 和 Flash 独立读图后取并集
  （`vlm.json` + `vlm_flash.json`）；主模型失败时 Flash 成功的结果仍会发布。
  但纯扫描件和纯 path 字样没有可供确定性文字通道兜底，两个模型都漏时仍会漏。
  `SCAN_ALL_PAGES=0` 保留选择性模式；在该模式下 `SCAN_NO_TEXT_PAGES=0` 还可连扫描页都关掉，
  这会明确产生 warning，且可能出现空结果。
- **plan 过滤是 fail-closed**。步骤4 只在 `view_type == "plan"` 的组框内保留放置。
  一页如果没有 plan 组（分类成了 section/detail，或步骤2 压根没给 view 组），
  结果就是**一个放置都不留**，不是「全页都留」。宁可少给，不要给错。
- **线型按 callout 末端绑定**。本地线型边车先聚类整页矢量线，再用 callout 箭头的
  末端选择它实际指到的线型；网页只发布当前引擎签名的结果，旧签名会显示为更新中，
  并由 `fence-linetype-refresh.timer` 低优先级重算，绝不会拿旧几何继续高亮。
- **空页会被隐藏**。`results.json` 的 `pages` 只包含有 fence 文字的页，页码是稀疏的
  1-based 字符串键；`page_count` 才是 PDF 的真实页数。
- **`sig_of` 只签 `(text, box_2d)`**。`label` / `tbl` 这类元数据变化**故意**不作废
  已付费的 raw。想让 symbol 步重新推理，改框或改文字，或者 bump `SYMBOL_PROMPT_V`。
