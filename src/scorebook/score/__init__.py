"""Score module for computing metrics on pre-computed outputs."""

from scorebook.score._async.score_async import score_async
from scorebook.score._sync.score import score

__all__ = ["score", "score_async"]
