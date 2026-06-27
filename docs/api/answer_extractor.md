# answer_extractor

Query-proximity answer extraction with regex patterns and optional LLM fallback.

`answer_extractor` keeps deterministic Python implementations for dates,
numbers, percentages, versions, names, yes/no answers, fuzzy matching, number
normalisation, and sentence selection. When the optional
`remanentia_answer_extractor` native extension is installed, the public
extractor helpers dispatch through the same typed callable contracts and retain
Python fallbacks for local-first deployments and test isolation.

The optional LLM helpers use a narrow backend protocol: a backend must provide
`complete(prompt, *, max_tokens, system="") -> str | None`.

::: answer_extractor.extract_answer

::: answer_extractor.extract_all_candidates
