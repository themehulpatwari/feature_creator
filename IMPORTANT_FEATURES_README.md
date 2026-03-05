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

# Feature Formulas

## Pairwise Task Formulas

### 3. Gaze Comparison Metrics

**gaze_comparison_reviewing_time_ratio**
```
left_engaged_time_s / (right_engaged_time_s + 0.001)
```

**gaze_comparison_reviewing_time_diff**
```
left_engaged_time_s - right_engaged_time_s
```

**gaze_comparison_reviewing_activity_ratio**
```
left_active_ratio / (right_active_ratio + 0.001)
```

**gaze_comparison_reviewing_activity_diff**
```
left_active_ratio - right_active_ratio
```

### 4. Mouse Comparison Metrics

**mouse_comparison_reviewing_time_ratio**
```
left_engaged_time_s / (right_engaged_time_s + 0.001)
```

**mouse_comparison_reviewing_time_diff**
```
left_engaged_time_s - right_engaged_time_s
```

**mouse_comparison_reviewing_activity_ratio**
```
left_active_ratio / (right_active_ratio + 0.001)
```

**mouse_comparison_reviewing_activity_diff**
```
left_active_ratio - right_active_ratio
```

Where:
- `engaged_time_s` = Sum of intervals < INACTIVITY_THRESHOLD_MS when looking at text / 1000
- `active_ratio` = engaged_time_s / reviewing_duration_s

### 5. Response-Specific Gaze Features (Windows 006, 007, 008, 014)

**{modality}_{side}_window_{NNN}**
```
mean(centre_idx / response_length) for data points in time window NNN
```

Window calculation:
```
time_span = max(rel_ts) - min(rel_ts)
window_size = time_span / 100
window_{i}_start = min(rel_ts) + i * window_size
window_{i}_end = min(rel_ts) + (i + 1) * window_size
```

### 6. Response-Specific Mouse Features (Windows 000, 022, 031, 033, 037, 039, 099)

Same formula as gaze windows (section 5).

**mouse_normalized_char_position_variance**
```
variance([centre_idx / response_length for all looking data points])
```

### 7. Other

**cross_modality_left_reviewing_duration_ratio_gaze_mouse**
```
gaze_left_reviewing_engaged_time_s / (mouse_left_reviewing_engaged_time_s + 0.001)
```

**response_B_response_length**
```
len(llm_response_2)
```

---

## Pointwise Task Formulas

### 3. Temporal Duration Metrics

**gaze_reviewing_duration_s**
```
(reviewing_end_timestamp - reviewing_start_timestamp) / 1000
```

**gaze_active_time_reviewing_s**
```
reviewing_duration_s * reviewing_active_ratio
```

Where:
```
reviewing_active_ratio = 1 - mean(is_not_looking) for reviewing phase data
```

**gaze_thinking_time_s**
```
composing_duration_s * composing_thinking_ratio
```

Where:
```
composing_thinking_ratio = count(on_screen AND not_at_text) / total_composing_points
```

**gaze_offscreen_time_composing_s**
```
composing_duration_s * composing_offscreen_ratio
```

Where:
```
composing_offscreen_ratio = mean(is_not_looking) for composing phase data
```

**mouse_offscreen_time_composing_s**
```
Same as gaze_offscreen_time_composing_s but for mouse data
```

### 4. Cross-Modality Metrics

**cross_modality_composing_activity_ratio_gaze_mouse**
```
gaze_composing_active_ratio / (mouse_composing_active_ratio + 0.001)
```

**mouse_reviewing_composing_activity_ratio**
```
mouse_reviewing_active_ratio / (mouse_composing_active_ratio + 0.001)
```

### 5. Gaze Window Features (20 windows)

**response_A_gaze_window_{NNN}**
```
mean(centre_idx / response_length) for data points in time window NNN
```

Window calculation same as pairwise (section 5).

### 6. Mouse Window Features (18 windows)

**response_A_mouse_window_{NNN}**
```
Same formula as gaze windows
```

**response_A_mouse_normalized_char_position_variance**
```
variance([centre_idx / response_length for all looking data points])
```

### 7. High-Level Engagement

**response_A_gaze_data_points**
```
count(all behavioral data points for this modality)
```

**response_A_gaze_normalized_avg_char_position**
```
mean([centre_idx / response_length for all looking data points])
```

---

## Key Constants

- **INACTIVITY_THRESHOLD_MS**: Maximum interval between consecutive points to count as continuous engagement (typically 1000-3000ms)
- **NUM_TIME_WINDOWS**: 100 (windows indexed 000-099)
- **EPSILON**: 0.001 (prevents division by zero in ratios)
- **MAX_RATIO**: 100.0 (caps extreme ratio values)

## Notes

- All ratios use safe division: `numerator / (denominator + EPSILON)`
- Ratios capped at MAX_RATIO to avoid extreme values
- Timestamps in milliseconds, durations converted to seconds
- Window features use normalized positions (0-1 scale) relative to response_length
