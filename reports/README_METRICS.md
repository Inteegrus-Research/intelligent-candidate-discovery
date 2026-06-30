# Pipeline Evaluation Report

Generated on: 2026-06-28 12:57:34

## Dataset
- Total candidates: **100,000**
- Retrieved: **1000**
- Final shortlist: **100**

## Retrieval
- Mean: 0.9536545061479343, Median: 0.9502790546647792, Std: 0.0241228441673862
- Range: [0.9179226953164424, 0.9999999999999999]

## Final Score
- Mean: 0.4848021468070445, Median: 0.44863022073432146, Std: 0.11980858240986109
- Range: [0.26437860603781493, 0.8583669976820792]

## Ranking Quality
- Top10 avg: 0.8108645352658771, Top100 avg: 0.7213343341158657
- Avg consecutive gap: 0.0018787242802839762

## Diversity
- Unique companies: 39
- Unique titles: 16
- Most common: rephrase.ai (8.0%)

## Candidate Archetypes
- **Best**: CAND_0039754 — Score 0.8584, Consistency 0.7365, Notice 30d
- **Median**: CAND_0064270 — Score 0.7071, Consistency 0.8531, Notice 45d
- **Lowest**: CAND_0083307 — Score 0.6724, Consistency 0.8382, Notice 120d

## Feature Group Contribution (Ablation Analysis)
- semantic: 41.8%
- career: 30.1%
- skill: 9.9%
- behavioral: 7.4%
- availability: 5.6%
- quality: 5.2%

## Explainability
- Coverage: 100.0%
- Avg length: 409.3 chars, 59.2 words

## Validation
- Overall: ✅ PASSED
- row_count_100: ✓
- unique_ids: ✓
- no_duplicates: ✓
- ranks_1_to_100: ✓
- score_monotonic: ✓
- reasoning_present: ✓
- ids_in_top1000: ✓
- score_in_range_0_1: ✓
- no_empty_reasoning: ✓

*All metrics from pipeline artifacts. No ground-truth labels required.*
