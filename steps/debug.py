"""可选的调试工件收集 —— 默认关闭，绝不改变正常输出.

用法：需要调试的调用传入 dbg = DebugSink()，步骤代码用 dbg.add()/dbg.set()
记录中间结果（裁剪窗口、每票的框、否决原因、被剥离的文字…），最后
dbg.data 随缓存 JSON 落盘（"debug" 键），前端"调试"开关按层渲染。
传 None（默认）时所有记录点都是零开销的 no-op。

坐标约定：所有落盘的框一律是页面帧 0-1000 [ymin,xmin,ymax,xmax]，
前端不需要知道任何裁剪细节。
"""


class DebugSink:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def add(self, key, value):
        self.data.setdefault(key, []).append(value)


def px_box_to_page(cx0, cy0, cw, ch, W, H):
    """像素裁剪窗口 → 页面帧 0-1000 框（调试显示用）."""
    return [round(cy0 / H * 1000, 1), round(cx0 / W * 1000, 1),
            round((cy0 + ch) / H * 1000, 1), round((cx0 + cw) / W * 1000, 1)]
