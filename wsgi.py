"""gunicorn 入口：`gunicorn -w 1 --threads 4 wsgi:app`.

webapp.py 的 ``__main__`` 块（Flask 自带服务器）在 gunicorn 下不会执行，
所以「让上次被强杀的作业从逐页 checkpoint 续跑」得在这里做一次。
故意不放进 webapp.py 的模块级：那样 import webapp 的单元测试会去动真实的
_jobs/ 目录。

只允许一个 worker（-w 1）：core.gemini.RECORDER 是进程级计费会话，
job._PROC_LOCK 也是进程级串行闸，多 worker 会让两者各算各的。
"""
import job
from webapp import app  # noqa: F401  (gunicorn 按名字找 app)

try:
    job.resume_interrupted()
except Exception as exc:                                        # noqa: BLE001
    print(f"[resume] skipped: {exc}", flush=True)
