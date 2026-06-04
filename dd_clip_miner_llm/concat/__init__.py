"""Smart concat pipeline.

重构后的实现：ConcatPipeline + Strategy 模式，支持 upfront health probe、
ProblemProfile（基于完整 ffmpeg 输出分类）、按输出驱动的 fallback 选择，
并保存每个 attempt 的完整日志到 concat_attempts/ 供调试。
"""

