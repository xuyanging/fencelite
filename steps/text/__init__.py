"""步骤1 找 fence 文字框 —— 三路来源，融合零遗漏.

  target      一处可编辑的检测目标 → 图片提示词 / 判词提示词两套固定脚手架
  vector      PDF 原生文字层 → 排版行级文字框（精确几何，本地免费）
  judge       项目去重字符串 → LLM 纯文本判词"是否 fence 相关"（缓存）
  vlm         VLM 读整页图找 fence 文字（单任务提示词；扫描页双模型 union）
  vlmcache    付费 raw 的内容哈希身份闸（PDF revision + 模型 + 提示词字节）
  clean       符号码剥离（marker码/嵌线token）+ VLM 框吸附收紧
  markers     剥符号码需要的矢量上下文（小闭合图形 + 直线段）
  merge       融合：vlm_items / vec_added / vec_covered（矢量兜底保证不丢）
  debug_view  纯本地的调试视图（每个矢量候选的来源与去向）
  page        一页的完整组装：fuse_page / vlm_needed

零遗漏保证：判词或关键词地板认可的每一行矢量文字，一定出现在融合结果里
（被 VLM 框覆盖 → vec_covered，没被覆盖 → vec_added）。
"""
from steps.text.judge import (KW, judge_candidates, judge_strings,  # noqa: F401
                              norm_text, prepare_judge_cache,
                              select_instances)
from steps.text.page import fuse_page, vlm_needed  # noqa: F401
from steps.text.target import (TARGET_DEFAULT, build_judge_prompt,  # noqa: F401
                               build_vlm_prompt, is_default_target)
from steps.text.vector import vector_scan, vector_scan_pages  # noqa: F401
from steps.text.vlm import scan_page, union_vlm  # noqa: F401
from steps.text.vlmcache import (SECONDARY_UNION_ROLE,  # noqa: F401
                                 is_current_primary_record,
                                 is_current_secondary_record,
                                 is_current_vlm_record, make_vlm_record,
                                 vlm_identity, vlm_identity_for_revision)
