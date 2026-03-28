# arcane_retriever

4-channel parallel retrieval with Reciprocal Rank Fusion (RRF).

Channels: FAST (BM25), WORKING (entity), DEEP (cross-session), TEMPORAL (date-aware). Includes a sufficiency loop that rewrites queries when results are insufficient.

::: arcane_retriever.ArcaneRetriever
    options:
      show_source: true
      members_order: source

::: arcane_retriever.RetrievalResult

::: arcane_retriever.FusedResult
