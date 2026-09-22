from .config import IndexMemConfig
from .scorer import RetentionScorer, RetentionScorerConfig

__all__ = ["IndexMemConfig", "IndexMemTextGenerationPipeline", "IndexMemInferenceContext"]


def __getattr__(name):
    if name == "IndexMemTextGenerationPipeline":
        from .pipeline import IndexMemTextGenerationPipeline

        return IndexMemTextGenerationPipeline
    if name == "IndexMemInferenceContext":
        from .inference.context import IndexMemInferenceContext

        return IndexMemInferenceContext
    raise AttributeError(name)
