# IT Career Insights Dashboard

A modern, interactive data analysis dashboard built with Python and Streamlit for analyzing student survey responses about interest in IT careers.

The app uses a small entrypoint wrapper in `analys.py`, while the full dashboard implementation lives in `dashboard_app.py`.

## 🎯 Features

### 1. **Data Processing**

- ✅ Automatic data loading from Google Sheets CSV export
- ✅ Multi-language column detection (Georgian & English)
- ✅ Smart missing value handling
- ✅ Multi-select answer parsing (split by semicolon/comma)
- ✅ Automatic percentage calculations

### 2. **Dashboard Structure**

#### 🗂️ Sidebar

- **Dataset source indicator** — "Uploaded CSV" or "Sample dataset" badge so you always know which data is active
- **Filter controls** — grade-level and gender multiselects backed by `st.session_state`; filters reset automatically when the data source changes
- **Reset filters** button — appears only when non-default selections are active
- **Dataset snapshot** — row count before and after filters
- **Export expander** — one-click download of the filtered CSV

#### 🔖 Active Filter Chips

- Coloured badge chips shown near the top of the main area whenever non-default filters are active

#### 📊 Top Section: Hero Banner + Key Takeaways

- Professional hero banner with high-level stats
- **Two key takeaways** displayed prominently as styled cards
- Full auto-generated insight list available under a collapsible *"View all insights"* expander

#### 📋 Data Quality Block

- Per-column missing-value counts for all required fields (colour-coded green/amber/red)
- Number of unique grades and genders visible in the current filtered view

#### 📈 KPI Section (Overview Cards)

- Total number of responses
- % interested in IT (with positive response count)
- % currently studying IT (with positive response count)
- % planning to choose IT career (with positive response count)

#### 📉 Visual Analysis Tabs

**📊 Overview Charts Tab:**

- Pie charts for single-choice questions (Interest, Gender)
- Horizontal bar charts for multi-choice questions with label truncation for long categories:
  - Most preferred IT fields
  - Motivations for choosing IT
  - Barriers to learning IT
  - Preferred learning methods
- Consistent empty-state message for charts with no data

**🔗 Relationships Tab:**

- Interest vs Studying Status (heatmap + percentage table)
- Interest vs Career Choice (heatmap + percentage table)
- Grade vs Interest (heatmap + percentage table)
- All displayed with colour-coded heatmaps and detailed tables

**🤖 Prediction Lab Tab:**

- Model info / disclaimer panel explaining limitations
- Three KPI cards: accuracy, training rows, input features
- Feature importance bar chart
- *Target class distribution* expander — bar chart showing career-choice balance in training data
- *Confusion matrix* expander — hold-out set predictions (appears when ≥ 2 classes)
- Styled interactive prediction form with a rich result card showing predicted class and confidence

**💬 Open Responses Tab:**

- Word frequency analysis from open-ended survey responses
- Bar chart showing most common words
- Full response data table for review

### 3. **Design Features**

- ✅ Dark, modern UI with refined spacing and component consistency
- ✅ Gradient background (deep navy tones)
- ✅ Professional hero banner
- ✅ Responsive layout (wide mode for large screens)
- ✅ Section cards with subtle shadows
- ✅ Interactive Plotly visualisations with consistent font/colour/height
- ✅ Georgian language support

### 4. **Code Structure**

`dashboard_app.py` is organised into focused render functions:

| Function | Responsibility |
|---|---|
| `render_sidebar()` | Source badge, filters, snapshot, export |
| `render_active_filters_summary()` | Filter chips in main area |
| `render_data_quality()` | Missing-value coverage block |
| `render_insights_section()` | Key takeaways + expander |
| `render_overview()` | Overview charts tab |
| `render_relationships()` | Relationships tab |
| `render_prediction_lab()` | Prediction Lab tab |
| `render_open_responses()` | Open Responses tab |

## 🚀 Getting Started

### Project Layout

- `analys.py` - Streamlit entrypoint wrapper
- `dashboard_app.py` - main dashboard logic, charts, filtering, and prediction model
- `requirements.txt` - runtime dependencies
- `.streamlit/config.toml` - Streamlit theme settings

### Installation

```bash
# Navigate to project directory
cd "path/to/dashboard"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS / Linux

# Install dependencies
pip install streamlit pandas plotly scikit-learn
```

### Running the Dashboard

```bash
streamlit run analys.py
```

You can also run the module directly with Streamlit if you prefer:

```bash
streamlit run dashboard_app.py
```

The dashboard will open at `http://localhost:8501` (or the next available port).

## ✅ Manual Test Checklist

| Step | Action | Expected result |
|---|---|---|
| 1 | Open app (no upload) | "Sample dataset" badge shown; sample data loads |
| 2 | Inspect sidebar filters | Grade & gender multiselects pre-filled; no Reset button |
| 3 | Change a filter | Reset button appears; active chips shown in main area |
| 4 | Click Reset filters | Chips disappear; all filters restored |
| 5 | Open Export expander | Download button present; click downloads CSV |
| 6 | Upload own CSV | "Uploaded CSV" badge appears; filters reset to new data |
| 7 | Select restrictive filter | Warning shown if no rows match |
| 8 | Overview tab | Charts render; empty-state message when series is empty |
| 9 | Relationships tab | Heatmaps and percentage tables shown |
| 10 | Prediction Lab tab | Model info panel visible; KPIs, feature importance shown; prediction form works; result card displayed |
| 11 | Open Responses tab | Word chart and response table shown (or info message if absent) |
| 12 | Data quality block | Missing-value counts visible below KPI cards |

## 📊 Data Format

The app expects a CSV export from Google Forms with columns like:

- Grade level (კლასი, grade, etc.)
- Gender (სქესი, gender, etc.)
- Interest in IT (ინტერესი, interest, etc.)
- IT direction/field (მიმართულება, field, etc.)
- Motivation (მოტივაცია, motivation, etc.)
- Learning methods (სწავლების მეთოდ, learning, etc.)
- Currently studying (სწავლობ, currently studying, etc.)
- Career choice (კარიერის აირჩევას, career, etc.)
- Barriers (ბარიერი, barriers, etc.)
- Open responses (აზრი, opinion, etc.)

## 🎨 Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Data Processing**: Pandas
- **Visualization**: Plotly
- **Machine Learning**: scikit-learn (DecisionTreeClassifier)
- **Data Source**: Google Sheets CSV export

## 📌 Key Insights Automatically Generated

The dashboard generates intelligent insights including:

- Overall student interest levels in IT
- Gap analysis between interest and actual studying
- Career intention trends
- Most popular IT fields
- Main motivations and barriers
- Grade-level interest patterns
- Conversion analysis (interested → studying → career)

## 🔧 Customization

You can modify:

- Color palette in `COLOR_SEQUENCE` variable
- CSS styling in `CSS` constant
- Column detection aliases in `COLUMN_ALIASES`
- Positive markers for yes/no detection in `POSITIVE_MARKERS`
- Model parameters in the `prepare_model()` function

## 📝 Notes

- The app caches data to improve performance
- Filters update all visualizations in real-time
- The ML model requires at least 6 balanced data points to train
- Filters are stored in `st.session_state` and reset automatically when the data source changes

- Multi-language support for Georgian and English column names

---

**Created**: April 2026  
**Status**: ✅ Production Ready
