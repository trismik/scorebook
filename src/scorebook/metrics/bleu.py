"""BLEU metric implementation for Scorebook, based on sacrebleu."""

from typing import Any, Dict, List, Tuple

import sacrebleu

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class BLEU(MetricBase):
    """BLEU metric implementation for Scorebook, based on sacrebleu."""

    def __init__(self, compact: bool = True, **kwargs: Any) -> None:
        """
        Generate BLEU metric.

        :param compact: if True, returns only the BLEU metric; if False,
        returns the full signature of BLEU.
        :param kwargs: additional arguments passed to BLEU.
        """

        self.compact = compact
        self.corpus_bleu = sacrebleu.metrics.BLEU(**kwargs)

        # Overwrite effective order for sentence level scores
        kwargs["effective_order"] = True
        self.sentence_bleu = sacrebleu.metrics.BLEU(**kwargs)

    def score(
        self, outputs: List[Any], labels: List[Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Calculate BLEU score between predictions and references.

        Args:
            outputs: A list of inference outputs.
            labels: A list of ground truth labels.

        Returns:
            The aggregate accuracy score for all items (correct predictions / total predictions).
            The item scores for each output-label pair (true/false).
        """

        if not outputs:  # Handle empty lists
            return {"BLEU": 0.0}, []

        item_scores = []
        # Calculate item scores
        for output, label in zip(outputs, labels):
            item_bleu: sacrebleu.metrics.BLEUScore = self.sentence_bleu.sentence_score(
                output, [label]
            )
            item_score = {
                "BLEU": item_bleu.score,
            }

            if not self.compact:
                item_score["1-gram"] = item_bleu.precisions[0]
                item_score["2-gram"] = item_bleu.precisions[1]
                item_score["3-gram"] = item_bleu.precisions[2]
                item_score["4-gram"] = item_bleu.precisions[3]
                item_score["BP"] = item_bleu.bp
                item_score["ratio"] = item_bleu.ratio
                item_score["hyp_len"] = item_bleu.sys_len
                item_score["ref_len"] = item_bleu.ref_len

            item_scores.append(item_score)

        # Calculate aggregate score

        corpus_bleu: sacrebleu.metrics.BLEUScore = self.corpus_bleu.corpus_score(outputs, [labels])
        aggregate_scores = {"BLEU": corpus_bleu.score}

        if not self.compact:
            aggregate_scores["1-gram"] = corpus_bleu.precisions[0]
            aggregate_scores["2-gram"] = corpus_bleu.precisions[1]
            aggregate_scores["3-gram"] = corpus_bleu.precisions[2]
            aggregate_scores["4-gram"] = corpus_bleu.precisions[3]
            aggregate_scores["BP"] = corpus_bleu.bp
            aggregate_scores["ratio"] = corpus_bleu.ratio
            aggregate_scores["hyp_len"] = corpus_bleu.sys_len
            aggregate_scores["ref_len"] = corpus_bleu.ref_len

        return aggregate_scores, item_scores
