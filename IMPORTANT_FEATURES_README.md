# Important Features Explanation

This document explains which features were selected as important and why.

## Overview

Two curated feature sets defined in [important_features.py](important_features.py):
- **PAIRWISE_IMPORTANT_FEATURES**: 61 features for predicting binary_preference
- **POINTWISE_IMPORTANT_FEATURES**: 56 features for predicting likert_1

**How they were selected**: Trained Random Forest and Logistic Regression models on all available features, then selected features with highest importance/coefficient values.

---

## Pairwise Important Features (61 features)

### Why These Features?
Selected based on Random Forest feature importance and Logistic Regression coefficient magnitudes from models trained on comprehensive feature set. These features had the strongest predictive power for determining which response users prefer.

### What They Include:

**1. LLM Model Identifiers (10)**
- Which LLM generated each response matters for preference
- Examples: `llm_1_claude-3-5-sonnet-20240620`, `llm_2_gpt-4o-mini`

**2. Session Context (7)**
- Metadata about the query interaction
- `adjustment`, `max_idx_left/right`, `query_length_right`, `response_left/right`, `total_entries_left/right`

**3. Gaze Comparison Metrics (4)**
- How users allocate attention between responses
- `gaze_comparison_reviewing_activity_ratio`, `gaze_comparison_reviewing_time_diff/ratio`

**4. Mouse Comparison Metrics (4)**
- Mouse interaction differences between responses
- `mouse_comparison_reviewing_activity_diff/ratio`, `mouse_comparison_reviewing_time_diff/ratio`

**5. Response-Specific Gaze Features (8)**
- Individual response attention patterns
- Engagement ratios, attention ratios, max char position, offscreen ratios
- Selected windows: 006, 007, 008, 014

**6. Response-Specific Mouse Features (10)**
- Individual response interaction patterns
- Engagement ratios, attention ratios, data points, normalized positions
- Selected windows: 000, 022, 031, 033, 037, 039, 099

**7. Other (2)**
- `cross_modality_left_reviewing_duration_ratio_gaze_mouse`
- `response_B_response_length`

### Key Pattern
**Comparison metrics are critical** - features that explicitly compare attention/activity between responses are most predictive for pairwise preference.

---

## Pointwise Important Features (56 features)

### Why These Features?
Selected based on Random Forest feature importance and Logistic Regression coefficient magnitudes from models trained on comprehensive feature set. These features had the strongest predictive power for determining single response quality ratings.

### What They Include:

**1. LLM Model Identifiers (5)**
- Which LLM generated the response
- Examples: `llm_1_claude-sonnet-4-5-20250929`, `llm_1_deepseek-ai/DeepSeek-V3`

**2. Session Context (3)**
- `adjustment`, `response_left`, `total_entries_left`, `camera_green`

**3. Temporal Duration Metrics (5)**
- How long users spend in different phases
- `gaze_active_time_reviewing_s`, `gaze_reviewing_duration_s`, `gaze_thinking_time_s`
- `gaze_offscreen_time_composing_s`, `mouse_offscreen_time_composing_s`

**4. Cross-Modality Metrics (2)**
- Coordination between gaze and mouse
- `cross_modality_composing_activity_ratio_gaze_mouse`
- `mouse_reviewing_composing_activity_ratio`

**5. Gaze Window Features (20)**
- Specific time slices of gaze attention (windows 001, 004, 006, 007, 009, 010, 014, 016, 018, 019, 023, 026, 038, 061, 067-069, 071, 073, 075, 092)
- Captures temporal sequence of attention

**6. Mouse Window Features (18)**
- Specific time slices of mouse interaction (windows 019, 021, 023, 026, 028, 034-035, 042, 047, 054-056, 072-073, 084, 098)
- Plus: `response_A_mouse_normalized_char_position_variance`

**7. High-Level Engagement (3)**
- `response_A_gaze_data_points`, `response_A_gaze_normalized_avg_char_position`

### Key Pattern
**Temporal patterns dominate** - many specific window features selected, showing that *when* users look/interact during the response is as important as aggregate metrics. Duration metrics also critical for pointwise quality.

---

## Why Different Features for Pairwise vs Pointwise?

| Aspect | Pairwise | Pointwise | Reason |
|--------|----------|-----------|--------|
| **Comparison metrics** | Heavy use | Not available | Pairwise needs relative comparison between responses |
| **Temporal durations** | Less important | Critical | Single response quality relates to time spent reviewing |
| **Window features** | Fewer (6) | More (38) | Pointwise relies more on temporal attention patterns |
| **LLM indicators** | 10 (both positions) | 5 (one position) | Pairwise captures model matchup effects |

---

## Summary

**Selection Method**: Train models on all features → Rank by importance/coefficients → Keep top predictive features

**Pairwise focuses on**: Comparison metrics, which response gets more attention, model matchups

**Pointwise focuses on**: Temporal patterns during single response review, duration of engagement phases, specific moments in reading sequence

Both retain LLM model information (which model generated responses) as this significantly affects predictions.
