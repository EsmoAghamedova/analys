# IT Career Insights Dashboard

A modern, interactive data analysis dashboard built with Python and Streamlit for analyzing student survey responses about interest in IT careers.

The app now uses a small entrypoint wrapper in `analys.py`, while the full dashboard implementation lives in `dashboard_app.py`.

## 🎯 Features

### 1. **Data Processing**

- ✅ Automatic data loading from Google Sheets CSV export
- ✅ Multi-language column detection (Georgian & English)
- ✅ Smart missing value handling
- ✅ Multi-select answer parsing (split by semicolon/comma)
- ✅ Automatic percentage calculations

### 2. **Dashboard Structure**

#### 📊 Top Section: Text Summary

- Professional natural-language insights automatically generated from data
- Includes: overall interest level, gaps between interest and action, popular fields, barriers
- Reads like a research conclusion (3-5 sentences)

#### 📈 KPI Section (Overview Cards)

- Total number of responses
- % interested in IT (with positive response count)
- % currently studying IT (with positive response count)
- % planning to choose IT career (with positive response count)

#### 📉 Visual Analysis Tabs

**Overview Charts Tab:**

- Pie charts for single-choice questions (Interest, Gender)
- Horizontal bar charts for multi-choice questions:
  - Most preferred IT fields
  - Motivations for choosing IT
  - Barriers to learning IT
  - Preferred learning methods

**Relationships Tab:**

- Interest vs Studying Status (heatmap + percentage table)
- Interest vs Career Choice (heatmap + percentage table)
- Grade vs Interest (heatmap + percentage table)
- All displayed with color-coded heatmaps and detailed tables

**Prediction Tab:**

- Machine Learning model (Decision Tree Classifier)
- Feature importance visualization
- Interactive prediction tool:
  - Select Interest level, Studying status, and Grade
  - Get predicted career choice with confidence percentage
- Model accuracy display

**Open Responses Tab:**

- Word frequency analysis from open-ended survey responses
- Bar chart showing most common words
- Full response data table for review

### 3. **Design Features**

- ✅ Minimal, modern UI with soft color palette
- ✅ Gradient background (light blue tones)
- ✅ Professional dark navy hero banner
- ✅ Responsive layout (wide mode for large screens)
- ✅ Clean spacing and section cards with subtle shadows
- ✅ Interactive Plotly visualizations
- ✅ Georgian language support

### 4. **Filtering Capabilities**

- Filter by Grade level (9, 10, 11, 12)
- Filter by Gender (Male/Female in Georgian)
- Real-time dashboard updates based on filters

### 5. **Machine Learning (Bonus)**

- Decision Tree Classification model
- Predicts career choice based on:
  - Interest in IT
  - Current studying status
  - Grade level
- Feature importance analysis
- Model accuracy metrics
- Interactive user input for predictions

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
.venv\Scripts\activate

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

The dashboard will open at `http://localhost:8501` (or the next available port)

## 📊 Data Format

The app expects a CSV export from Google Forms with columns like:

- Grade level (კლასი, grade, etc.)
- Gender (სქესი, gender, etc.)
- Interest in IT (ინტერესი, interest, etc.)
- IT direction/field (მიმართულება, field, etc.)
- Motivation (მოტივაცია, motivation, etc.)
- Learning methods (სწავლების მეთოდ, learning, etc.)
- Currently studying (სწავლობ, studying, etc.)
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
- CSS styling in `SOFT_TEMPLATE`
- Column detection aliases in `COLUMN_ALIASES`
- Positive markers for yes/no detection in `POSITIVE_MARKERS`
- Model parameters in the `prepare_model()` function

## 📝 Notes

- The app caches data to improve performance
- Filters update all visualizations in real-time
- The ML model requires at least 6 balanced data points to train
- Multi-language support for Georgian and English column names

---

**Created**: April 2026  
**Status**: ✅ Production Ready
