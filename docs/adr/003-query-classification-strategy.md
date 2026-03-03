# ADR 003: Query Classification Strategy

**Status**: Accepted  
**Date**: 2026-03-03  
**Decision Maker**: shankssc

## Context

Need to route queries to specialized retrieval/generation strategies based on intent to:

- Improve answer quality (factual queries get citations, recommendations get diversity)
- Reduce LLM costs (avoid creative generation for simple factual queries)
- Enable future personalization (user preferences per intent type)

## Decision

Implement rule-based classifier (Phase 2) with migration path to ML-based (Phase 5).

### Rule-Based Approach (Phase 2)

- Regex pattern matching with confidence scoring
- Priority cascade: recommendation → comparison → creative → meta → factual
- Negative adjustments for uncertainty language (`I think`, `maybe`, etc.)
- Defense-in-depth validation (type → content → length)

### ML-Based Future (Phase 5)

- Fine-tuned transformer on 1,000+ labeled anime queries
- Target: >95% accuracy (vs 85% rule-based baseline)
- Active learning for pattern discovery

## Consequences

✅ **Positive**

- Zero inference latency (<1ms classification)
- Full interpretability (matched patterns logged)
- No training data required for Phase 2 launch
- Easy debugging and pattern updates

⚠️ **Negative**

- Limited to predefined patterns (may miss nuanced queries)
- Requires manual pattern updates for new query types
- Accuracy ceiling (~85%) vs ML approach (>95%)

## Validation

- 25+ intent classification tests pass
- Edge cases handled (empty/whitespace/short queries)
- Confidence adjustments verified for uncertainty language
- 100% unit test pass rate
