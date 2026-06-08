"""Smart concat pipeline.

模块结构：
- health.py      — 拼接前 H.264/HEVC 健康探测、TS transmux
- strategies.py  — 6 种 Strategy（copy / mkvmerge / discard-corrupt / repair / …）
- runner.py      — ConcatPipeline 主流程、pre-sanitize、策略调度
- helpers.py     — 候选输出、legacy 辅助函数、normalize/reencode 实现
- pipeline.py    — 对外兼容 API（re-export）
"""

