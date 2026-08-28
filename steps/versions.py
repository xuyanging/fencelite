"""缓存失效开关 —— 全项目唯一一处.

改任何一个都会让对应缓存重算；带「付费」标记的会重发 Gemini 调用。
数值故意与生产 5051 保持一致，这样 tools/import_project.py 导进来的旧缓存
仍然算当期，不必重花钱。
"""

# results.json 的融合语义（vlm_items / vec_added / vec_covered 三桶口径）。
# bump 后从已缓存的 vec / 判词 / VLM raw 免费重建。
FUSED_VERSION = 2

# textjudge.json 的判词语义。无 v 字段的旧缓存视为 1。
# bump = 重判所有字符串（付费，但纯文本、最便宜的一档）。
#   2：判词里明确「纯编码串（4CL / 6DMP / SF）是标记标签，不是描述，一律不 flag」
#      —— 之前 "4CL" 被判成 fence 相关（语义上没错），于是平面图上每个标记都
#      成了一条独立文字项，把真正的 callout 埋掉了。
TEXT_JUDGE_VERSION = 2

# 图例 group+symbol 的提示词 / response schema 版本。
# bump = 每页重新付费推理。改 steps/prompts.py 必须同步 bump。
#   17：给 TASK B 补了两块硬要求 ——「每个样例单独、紧贴地框」（实测模型会把
#       整列样例套用同一个列状框，九行全部偏左、三行还纵向错位；并明确要求
#       shape 的框必须把它读出来的 value 字符包在里面），以及「逐行走完整个
#       图例块，别静默跳过小标记」；OCR 易混对补上 8/6、F/P、M/N（实测把
#       图上的 8DMF 读成了 6DMP）。
SYMBOL_PROMPT_V = 17

# 图例 symbol 的发布过滤 schema（owner 硬校验 + 图例组内硬校验 + 同框去重）。
# 只要 SYMBOL_PROMPT_V 不变，bump 就是拿已存的 raw 免费重过滤。
#   19（本版）改了两处发布语义，所以必须 bump：
#     * 组内闸改成**几何为准** —— 框中心必须落在 legend/schedule/note_cluster
#       组框内（±2）。模型自称的 group_index 不再是通行证（老口径是「group_index
#       命中 **或** 几何命中」的或关系，平面图里的 marker 只要 group_index 填对
#       就能发布），降级成审计字段 claimed_group_index；框中心不落在任何组框里
#       时退回按 owner 文字的位置判断。
#     * 同框去重 —— 同一个样例框被配给多个文字时只发布一条（裸编码 "4CL" 漏进
#       文字层导致的重复），优先留「与框重叠度低 + 文字更长」的描述行。
SYMBOL_VERSION = 19

# 视图投影分类器（plan / elevation / section / detail / other）。
# 独立缓存：bump 只重付分类，不动 symbol raw。
VIEW_VERSION = 1

# 本地 shape 放置匹配 + plan 视图过滤。纯几何、零模型调用，bump 免费重算。
#   3（本版）：放置框补上标记的外圈（core/symbolmatch.py 的 _enclosing_outline）。
#     发布的框原来是「实际匹配上的图元的并集」，而编号标记（圆圈里一个数字）
#     常常只匹配上里面的数字、圈没进去 —— 实测 combined_bid P20 七个符号的
#     放置框高度一律只有 12.1pt，而真实圆圈是 18.1pt，即只框住 67%，界面上
#     看就是框画歪了。判据锚在模板上（模板就是图例样例、也就是整个标记）：
#     候选外框必须完整包住已匹配的组，且不得超过模板尺寸的 OUTLINE_TOL 倍。
#     这样既捡回圆圈，又不会把标记所在的围栏长线、建筑外轮廓一起吞进来。
#     实测：放置框 7.8~15.6x12.1pt → 18.1x17.3~19.0pt（真实圆圈 18.1x18.1pt），
#     且**匹配数量一个没变**，只改框的范围、不改匹配结果。
#   2：匹配之前先用矢量层把样例框校准（steps/snap_boxes.py）——
#     模型在整页图上给的框会漂（实测 drawings_volume_4_binder P4 九个样例整列
#     偏左 ~10 单位、两个纵向偏 6 单位），拿漂了的框当模板去匹配自然也差；
#     顺带产出「图例行文字框裁掉行首编码」的 text_trim 表。
PLACEMENT_VERSION = 3
