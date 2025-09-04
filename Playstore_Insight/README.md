# Agentic AI Trend Analysis System

### Assignment Overview

Build an AI agent that processes Google Play Store reviews from June 2024 onwards and generates trend analysis reports for issues, requests, and feedback.

### System Architecture

#### Multi-Agent Architecture

1. **Discovery Agent**: Rule-based + Clustering + N-gram analysis
2. **Classification Agent**: High-recall review classification
3. **Consolidation Agent**: 85% similarity threshold merging
4. **Quality Assessment Agent**: Topic quality validation

#### Key Features

- **Intelligent Consolidation**: Merges similar topics like:
  - "Delivery guy was rude" → "Delivery partner rude"
  - "Delivery partner behaved badly" → "Delivery partner rude"
  - "Delivery person was impolite" → "Delivery partner rude"

### Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run assignment mode
python main_pipeline_agentic.py --mode assignment \
  --app-link "https://play.google.com/store/apps/details?id=com.swiggy.android" \
  --date "2024-06-30"
```

### Usage Examples

#### Assignment Input/Output Format

```bash
# Input: App store link and target date
python main_pipeline_agentic.py --mode assignment
  --app-link "https://play.google.com/store/apps/details?id=com.swiggy.android"
  --date "2024-06-30"

# Output: Trend table (Topics × Dates × Frequency)
```

#### Demo Mode

```bash
python main_pipeline_agentic.py --mode demo
```

### Output Structure

```
/output/
├── comprehensive_trend_report_YYYY-MM-DD_HHMMSS.csv
└── comprehensive_trend_report_YYYY-MM-DD_HHMMSS.html
```

### Report Format

- **Rows**: Topics (delivery issues, food quality, app performance, etc.)
- **Columns**: Dates from T to T-30
- **Cells**: Frequency of topic occurrence

### Technical Implementation

#### Consolidation Algorithm

- Lexical similarity (word overlap, edit distance)
- Semantic similarity (domain taxonomy)
- Vector similarity (TF-IDF cosine)
- Context similarity (confidence, method, support)
- **Threshold**: 85% for conservative merging

#### High-Recall Features

- Multiple discovery methods per agent
- Semantic pattern matching
- Emerging topic detection via n-grams
- Advanced similarity measures

### File Structure

```
├── src/
│   ├── database_setup.py              # SQLite database management
│   ├── data_ingestion.py              # Google Play Store scraping
│   ├── agentic_topic_detection.py     # Multi-agent topic detection
│   ├── topic_consolidation.py         # Advanced consolidation engine
│   └── trend_generator.py             # Professional trend reports
├── main_pipeline_agentic.py           # Main pipeline orchestration
├── output/                            # Generated reports
├── requirements.txt                   # Dependencies
├── demo_script.md                     # Video demonstration guide
└── README.md                          # This file
```
