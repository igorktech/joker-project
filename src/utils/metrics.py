import sacrebleu.metrics as sbmetrics
from evaluate import load


# Sacrebleu metrics
bleu_calc = sbmetrics.BLEU()
chrf_calc = sbmetrics.CHRF(word_order=2)
# WMT22 ensemble metric
comet22_metric = load(
    "comet",
    module_type="metric",
    model_id="Unbabel/wmt22-comet-da"
)

def compute_translation_metrics(
    preds,
    refs,
    language_pairs=None,
    prompts=None,
    comet_metric=None,
):
    """
    Compute BLEU, chrF++ and optional COMET metrics for cleaned predictions and references.

    Args:
        preds: List of model-generated translations (EOS-stripped, cleaned)
        refs: List of lists of reference translations (EOS-stripped, cleaned)
        language_pairs: Optional list of source-target codes for per-language metrics
        prompts: Optional list of source prompts (required for COMET)
        comet_metric: Optional sacreCOMET metric instance

    Returns:
        Dict of metric names to scores
    """
    # Corpus-level BLEU and chrF++
    metrics = {
        'bleu': bleu_calc.corpus_score(preds, refs).score,
        'chrf++': chrf_calc.corpus_score(preds, refs).score,
    }

    # COMET if provided
    if comet_metric is not None and prompts is not None:
        comet_out = comet_metric.compute(
            sources=prompts,
            predictions=preds,
            references=[r[0] for r in refs]
        )
        metrics['comet'] = comet_out.get('mean_score')

    # Per-language breakdown
    if language_pairs is not None:
        lang_group = {}
        for idx, pair in enumerate(language_pairs):
            tgt = pair.split('-')[-1]
            group = lang_group.setdefault(tgt, {'preds': [], 'refs': []})
            group['preds'].append(preds[idx])
            group['refs'].append(refs[idx])
        for lang, data in lang_group.items():
            metrics[f'bleu_{lang}'] = bleu_calc.corpus_score(data['preds'], data['refs']).score
            metrics[f'chrf++_{lang}'] = chrf_calc.corpus_score(data['preds'], data['refs']).score

    return metrics