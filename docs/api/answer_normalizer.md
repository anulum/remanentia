# answer_normalizer

Post-processing for extracted answers: hedging removal, yes/no polarity matching, list overlap scoring, semantic similarity comparison.

The Python surface keeps a typed fallback path for environments without the
optional `remanentia_answer_normalizer` native extension. When local embedding
models are unavailable, `semantic_similarity` falls back to normalized lexical
similarity instead of failing the answer-comparison pipeline.

::: answer_normalizer
    options:
      show_source: true
      members_order: source
