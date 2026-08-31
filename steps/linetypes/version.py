"""线型层的缓存失效开关 —— 只此一处.

故意**不**放进 steps/versions.py：那张表是四个既有付费步的开关，线型层是一个
独立、免费、可整体关掉的新模块，版本号跟着模块自己走（与 steps/arrows.py 的
ARROWS_VERSION 同一个做法）。

bump 的代价：🆓 纯本地几何 + 一次边车子进程，零模型调用。但**不便宜** ——
边车实测一页要 100 s 以上（source-aligned 抽取 ~15 s + 聚类 ~95 s，gladstone P2
的 3563 个 op），所以 bump 会让盘上所有项目的线型层同时停止发布，要逐个重跑
才回来。改之前先想清楚。

  6（本版）：symbol 矢量放置即使没有引线，也以 placement 框中心查询最近的已知
     线型；边车先排除框周围属于 marker 自己的局部图元，避免把 callout 的边框、
     字码或短刺再次识别成目标线。相同 symbol 的多个 placement 以共同可达候选
     形成共识，可以压过单个 placement 上误猜出来的伪引线；纯 gate 不参与，
     同时包含 fence 与 gate 的文字继续按 fence 优先。
  5：高亮范围的裁剪单位从「引擎 group」换成「**几何连通走线**」。
     v3 按 group 裁是为了修 lenexa P4 的高亮过多，但它在 gladstone P4 上把
     线型 #5 的第三个 group（中间那条横向分隔围栏，46 op / 305 段）裁掉了 ——
     那明明是同一道围栏的连续段。实测精确几何距离：gladstone 的真实续接是
     20↔51 = 0.0200、51↔52 = 0.1349 页帧单位（贴着），而 lenexa 那条无关的
     左侧长带 41↔74 = 1.1496（只是靠近、没接触），差近 10 倍。所以判据改成
     「接触即同一条走线」，容差 RUN_TOUCH_PT = 0.5 pt（IR 帧，等向）。
     多个末端指到不同走线时取并集。同型但没接上的走线进 dropped_runs 报出来。
     顺带：每个末端上报 nearest_point（页面帧），前端画「末端→线」的判据线段。
  4：**不再漏线型**。边车改成一次取 method1/method2/fused 三份输出，
     并把「method1 认领 / method2 从未认领 / fused 却无主」的覆盖补回来。
     起因：fusion 的 method2-owns-overlap-v1 是按 **global 类型整体**裁决的，
     method2 只要赢下某个成员组，整个跨组类型就被溶解，连它在别的组里、
     method2 从未碰过的部分一起消失。实测 gladstone P4：method1 1772 op、
     method2 186（且是 method1 真子集）、fused 只剩 897 —— 875 个**无竞争者**
     的 op 凭空没了，其中 group 57 那条 58-op 的线 method2 一个 op 都没碰过；
     这 875 个里 99.1% 距离任何 method2 op 超过 36 pt（中位 517.7 pt），
     是图上完全不同的部位。补回后 897+875=1772，与 method1∪method2 相等。
     补回的类型带 recovered_from_fusion=True，可分辨、可审计。
     同时 cpu_budget 默认 1→16（实测 sheet 2/4 上 1 与 16 结果逐项相同，
     100.4s→71.0s / 23.6s→13.8s），并把它计入缓存签名。
     **与 TS 的对齐不受影响**：fused 部分仍与冻结的 TS r10/r46 逐 op 一致，
     补回的只是 TS 侧同样会丢、但本就属于 method1 的覆盖。
  3：边车按**引擎 group** 分桶交付几何，高亮只画「被这个 callout 的
     末端指到的 group」。一个 global 线型是若干 group 的局部簇按签名相似度
     跨组合并出来的：lenexa P4 的 #5 由 group 50/41/73/74 四块并成
     （min_sim 0.9592，四块空间上完全分开），而 callout 只指到 group 41
     那条波浪线；照整个 global 画就把右上角一大堆和左边两条虚线带一起点亮了。
     限制到 group 41 之后是 134/607 个 op、736/1237 段。
  2：三处判据变更。
     * 输入换成 **source-aligned** PageIR（source_aligned_page_ir_from_pdf_path），
       不再用 pdf_adapter 那条。source_page_adapter.py:1-10 明写 source_content
       才是 authored paint order / path topology / style 的权威，pdf_adapter 的
       path 列表"只作为诊断计数保留"；而 pdf_adapter 从不设
       source_provenance_exact（ir.py:131 默认 False），grouping.py 的真实
       拆分/合并拿它当闸。实测同一页 plain 8 个线型 / residual 2852 vs
       aligned 12 个 / residual 2103，type_uid 只重合 2 个。
     * 绑定判据从「最近的簇」改成「**拥有离 tip 最近那条 path op 的簇**」，
       最近的 op 是 residual 时给出 residual 而不是硬凑一个近的簇。实测一页
       只有 38% 的 path op 属于任何簇，老判据会稳定地给出看着对的错高亮。
     * 先剔掉 callout 自己的引线与箭头再比距离（重复的箭头本身就是
       compound_path_periodic 会认的东西）。
  1：首版（已废弃的判据，见上）。
"""

LINETYPE_VERSION = 6
