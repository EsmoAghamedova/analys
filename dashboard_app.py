from __future__ import annotations

import re
from collections import Counter
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


APP_TITLE = "IT Career Insights Dashboard"
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRMg6F5hV70961sWNCJ6CqieLsDmunxQJ_726xmcyKlRQUzDuEI8X3K3AK5U7QFSnCUmi-jwwO0VWo1/pub?output=csv"

COLUMN_ALIASES = {
    "grade": ["კლასი", "მერამდენე კლასში ხარ", "grade"],
    "gender": ["სქესი", "gender"],
    "interest": ["ინტერესი", "it სფეროების მიმართ", "interest"],
    "field": ["მიმართულება", "preferred", "direction", "it field"],
    "motivation": ["მოტივაცია", "რატომ აირჩიე", "motivation"],
    "learning": ["სწავლების", "learning methods", "როგორ სწავლობ", "მეთოდ"],
    "studying": ["სწავლობ", "currently studying", "ამჟამად IT"],
    "career": ["კარიერის", "career", "აირჩევას"],
    "barriers": ["ბარიერ", "barriers", "სირთულ"],
    "opinion": ["აზრი", "opinion", "comment"],
}

REQUIRED_COLUMNS = ["grade", "gender", "interest", "studying", "career"]

POSITIVE_MARKERS = {
    "yes": ["დიახ", "კი", "yes", "sure", "interested", "plan"],
}

COLOR_SEQUENCE = [
    "#2563eb",
    "#7c3aed",
    "#10b981",
    "#f59e0b",
    "#ec4899",
    "#06b6d4",
    "#8b5cf6",
    "#0f766e",
]

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Georgian:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --panel: rgba(13, 24, 43, 0.78);
        --panel-strong: #0d1d33;
        --border: rgba(148, 163, 184, 0.18);
        --text: #e7eef9;
        --muted: #91a4c4;
        --accent: #60a5fa;
        --accent-2: #8b5cf6;
        --accent-3: #14b8a6;
    }

    html, body, .stApp {
        font-family: "Noto Sans Georgian", sans-serif;
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.2), transparent 28%),
            radial-gradient(circle at bottom center, rgba(20, 184, 166, 0.18), transparent 32%),
            linear-gradient(180deg, #06101c 0%, #0a1628 45%, #07111f 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero-shell {
        background: linear-gradient(135deg, rgba(13, 29, 51, 0.95), rgba(10, 20, 36, 0.88));
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 24px 80px rgba(2, 6, 23, 0.45);
        margin-bottom: 1.5rem;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(96, 165, 250, 0.12);
        color: #c9ddff;
        border: 1px solid rgba(96, 165, 250, 0.22);
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.5fr 0.9fr;
        gap: 24px;
        align-items: stretch;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 3.55rem);
        line-height: 1.06;
        margin: 0 0 14px 0;
        letter-spacing: -0.03em;
        color: #f4f8ff;
    }

    .hero-copy {
        color: #b6c7e2;
        font-size: 1.02rem;
        line-height: 1.75;
        max-width: 60rem;
        margin-bottom: 16px;
    }

    .hero-panel {
        background: rgba(8, 15, 28, 0.55);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 18px;
        display: grid;
        gap: 12px;
        align-content: start;
    }

    .hero-stat {
        padding: 14px 16px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.17), rgba(124, 58, 237, 0.14));
        border: 1px solid rgba(148, 163, 184, 0.14);
    }

    .hero-stat strong {
        display: block;
        color: #f6fbff;
        font-size: 1.4rem;
        line-height: 1.15;
    }

    .hero-stat span {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .metric-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, rgba(12,25,42,0.86), rgba(8,16,29,0.86));
        border: 1px solid rgba(148, 163, 184, 0.06);
        border-radius: 20px;
        padding: 18px 18px 16px;
        min-height: 132px;
        box-shadow: 0 10px 28px rgba(2, 6, 23, 0.22);
        transition: transform 200ms ease, box-shadow 200ms ease;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 6px;
        background: linear-gradient(180deg, var(--accent), var(--accent-2));
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
    }

    .metric-label {
        color: #9fb3d2;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2.25rem;
        line-height: 1.05;
        font-weight: 800;
        color: #f7fbff;
        margin-bottom: 8px;
        letter-spacing: -0.01em;
    }

    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 22px 60px rgba(2, 6, 23, 0.34);
    }

    .metric-caption {
        color: #bfd0ea;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .insight-card {
        background: linear-gradient(180deg, rgba(12, 25, 42, 0.95), rgba(8, 16, 29, 0.94));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 22px;
        margin: 1.3rem 0 1.2rem;
        box-shadow: 0 18px 48px rgba(2, 6, 23, 0.24);
    }

    .insight-card h3 {
        margin: 0 0 10px 0;
        color: #f5faff;
        font-size: 1.1rem;
    }

    .insight-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #d3def1;
        line-height: 1.75;
    }

    .section-shell {
        background: rgba(10, 20, 36, 0.42);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 22px;
        padding: 20px;
        margin-top: 18px;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: #f5faff;
        letter-spacing: -0.02em;
    }

    .section-subtitle {
        color: var(--muted);
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(9, 18, 31, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 0.25rem;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a9c8;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: #eff6ff;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.34), rgba(124, 58, 237, 0.34));
    }

    div[data-testid="stSidebar"] {
        background: rgba(7, 17, 31, 0.92);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    div[data-testid="stMetric"] {
        background: transparent;
    }

    .stDataFrame, .stPlotlyChart {
        border-radius: 16px;
    }

    .mini-note {
        color: #9fb3d2;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* ── Source badge ─────────────────────────────────── */
    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .source-uploaded {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6ee7b7;
    }
    .source-sample {
        background: rgba(96, 165, 250, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.28);
        color: #93c5fd;
    }

    /* ── Active-filter chips ──────────────────────────── */
    .chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin: 0.5rem 0 1rem;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.18);
        border: 1px solid rgba(96, 165, 250, 0.3);
        color: #93c5fd;
        font-size: 0.78rem;
        white-space: nowrap;
    }
    .chip-label {
        color: var(--muted);
        font-size: 0.75rem;
        margin-right: 2px;
    }
    .chips-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.5rem;
    }

    /* ── Key takeaways ────────────────────────────────── */
    .takeaway-wrap {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 1rem 0 0.5rem;
    }
    @media (max-width: 640px) {
        .takeaway-wrap { grid-template-columns: 1fr; }
    }
    .takeaway-card {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(124, 58, 237, 0.11));
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 16px;
        padding: 16px 18px;
    }
    .takeaway-label {
        color: #60a5fa;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 6px;
    }
    .takeaway-text {
        color: #e2ecff;
        font-size: 0.97rem;
        line-height: 1.65;
    }

    /* ── Data-quality block ───────────────────────────── */
    .dq-block {
        background: rgba(10, 20, 36, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 14px 18px;
        margin: 1rem 0;
    }
    .dq-title {
        color: #94a9c8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }
    .dq-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 8px;
    }
    .dq-item {
        background: rgba(6, 15, 28, 0.55);
        border-radius: 10px;
        padding: 8px 12px;
    }
    .dq-item-label {
        color: var(--muted);
        font-size: 0.76rem;
        margin-bottom: 2px;
    }
    .dq-item-val { font-size: 0.95rem; font-weight: 600; }
    .dq-good  { color: #34d399; }
    .dq-warn  { color: #fbbf24; }
    .dq-bad   { color: #f87171; }

    /* ── Prediction result card ───────────────────────── */
    .pred-result {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(20, 184, 166, 0.11));
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 18px;
        padding: 22px 24px;
        margin-top: 16px;
        text-align: center;
    }
    .pred-result-label {
        color: #6ee7b7;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 8px;
    }
    .pred-result-value {
        color: #f0fdf4;
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .pred-result-conf {
        color: #86efac;
        font-size: 0.95rem;
    }

    /* ── Model info panel ─────────────────────────────── */
    .model-info {
        background: rgba(10, 20, 36, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 1rem;
        color: #9fb3d2;
        font-size: 0.88rem;
        line-height: 1.7;
    }
    .model-info strong { color: #c8d6ea; }

    /* ── Empty-chart annotation ───────────────────────── */
    .empty-chart-msg {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 200px;
        border: 1px dashed rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        color: var(--muted);
        font-size: 0.9rem;
    }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def canonical_text(value: object) -> str:
    text = norm_text(value).lower()
    return re.sub(r"[^\w\s]+", "", text, flags=re.UNICODE)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [norm_text(column) for column in cleaned.columns]

    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].astype(str).map(norm_text)
            cleaned[column] = cleaned[column].replace(
                {"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return cleaned


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    available = list(frame.columns)
    resolved: dict[str, str] = {}
    canonical_lookup = {canonical_text(column): column for column in available}

    for key, aliases in COLUMN_ALIASES.items():
        match = None

        for alias in aliases:
            alias_norm = norm_text(alias)
            alias_canon = canonical_text(alias_norm)

            if alias_norm in available:
                match = alias_norm
                break
            if alias_canon in canonical_lookup:
                match = canonical_lookup[alias_canon]
                break

        if match is None:
            tokens = [token for token in canonical_text(
                aliases[0]).split() if token]
            for column in available:
                column_canon = canonical_text(column)
                if tokens and all(token in column_canon for token in tokens):
                    match = column
                    break

        if match:
            resolved[key] = match

    missing = [key for key in REQUIRED_COLUMNS if key not in resolved]
    if missing:
        raise KeyError(
            "Could not identify required columns: "
            + ", ".join(missing)
            + f". Available columns: {available}"
        )

    return resolved


def split_multiselect(series: pd.Series) -> pd.Series:
    values: list[str] = []
    for cell in series.dropna():
        for item in re.split(r";|,|\n", str(cell)):
            cleaned = norm_text(item)
            if cleaned:
                values.append(cleaned)

    if not values:
        return pd.Series(dtype=int)

    return pd.Series(values).value_counts().sort_values(ascending=False)


def percent(series: pd.Series, positive_markers: list[str]) -> tuple[int, float]:
    values = series.dropna().astype(str)
    if values.empty:
        return 0, 0.0

    total = len(values)
    positive = values.map(lambda value: any(
        marker in canonical_text(value) for marker in positive_markers)).sum()
    return int(positive), round(positive / total * 100, 1)


def pct_table(frame: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    table = pd.crosstab(frame[row_col], frame[col_col],
                        normalize="index") * 100
    return table.round(1)


def format_pct_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table

    def fmt_cell(value):
        try:
            if pd.isna(value):
                return ""
            return f"{float(value):.1f}%"
        except Exception:
            return value

    return table.copy().apply(lambda col: col.map(fmt_cell))


def make_figure_base(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=8, r=8, t=54, b=8),
        title_font_size=16,
        title_font_color="#eff6ff",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d6ea", size=12),
        legend_title_text="",
    )
    return fig


def make_empty_figure(title: str, height: int = 320) -> go.Figure:
    """Return a styled placeholder figure when there is no data to display."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available for the current filter selection",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13, color="#91a4c4"),
    )
    return make_figure_base(fig, title, height=height)


_MAX_LABEL_LEN = 32


def _truncate_label(text: str) -> str:
    return text[: _MAX_LABEL_LEN - 1] + "…" if len(text) > _MAX_LABEL_LEN else text


def make_horizontal_bar(data: pd.Series, title: str, color_index: int = 1) -> go.Figure:
    if data.empty:
        return make_empty_figure(title, height=320)

    plot_df = data.reset_index()
    plot_df.columns = ["label", "count"]
    plot_df["label"] = plot_df["label"].astype(str).map(_truncate_label)
    plot_df = plot_df.sort_values("count", ascending=True)

    fig = px.bar(
        plot_df,
        x="count",
        y="label",
        orientation="h",
        text="count",
        color_discrete_sequence=[
            COLOR_SEQUENCE[color_index % len(COLOR_SEQUENCE)]],
        title=title,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
    )
    return make_figure_base(fig, title, height=max(300, 60 + len(plot_df) * 36))


def make_pie(series: pd.Series, title: str, color_sequence: list[str]) -> go.Figure:
    counts = series.dropna().value_counts()
    if counts.empty:
        return make_empty_figure(title, height=320)

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title=title,
        color_discrete_sequence=color_sequence,
        hole=0.48,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return make_figure_base(fig, title, height=330)


def make_heatmap(table: pd.DataFrame, title: str) -> go.Figure:
    if table.empty:
        return make_empty_figure(title, height=320)

    fig = go.Figure(
        data=go.Heatmap(
            z=table.values,
            x=list(table.columns),
            y=list(table.index),
            colorscale=[[0, "#0f172a"], [0.35, "#2563eb"],
                        [0.7, "#7c3aed"], [1, "#14b8a6"]],
            text=table.values,
            hovertemplate="%{y} / %{x}<br>%{z:.1f}%<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
    )
    return make_figure_base(fig, title, height=340)


def class_to_str(value: object) -> str:
    text = norm_text(value)
    return text or "უცნობი"


@st.cache_data(show_spinner=False)
def load_data(uploaded_file_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_file_bytes is not None:
        return clean_frame(pd.read_csv(BytesIO(uploaded_file_bytes)))
    return clean_frame(pd.read_csv(DATA_URL))


def build_insights(frame: pd.DataFrame, cols: dict[str, str]) -> list[str]:
    total = len(frame)
    if total == 0:
        return ["No responses are available after filtering."]

    interest_yes, interest_pct = percent(
        frame[cols["interest"]], POSITIVE_MARKERS["yes"])
    studying_yes, studying_pct = percent(
        frame[cols["studying"]], POSITIVE_MARKERS["yes"])
    career_yes, career_pct = percent(
        frame[cols["career"]], POSITIVE_MARKERS["yes"])

    field_counts = split_multiselect(
        frame[cols["field"]]) if "field" in cols else pd.Series(dtype=int)
    barrier_counts = split_multiselect(
        frame[cols["barriers"]]) if "barriers" in cols else pd.Series(dtype=int)

    interest_to_study = pct_table(frame, cols["interest"], cols["studying"])
    grade_to_interest = pct_table(frame, cols["grade"], cols["interest"])

    sentences: list[str] = []
    sentences.append(
        f"From {total} responses, {interest_pct}% show interest in IT, {studying_pct}% are currently studying it, and {career_pct}% plan to choose an IT career."
    )

    if interest_pct > studying_pct + 10:
        sentences.append(
            f"There is a clear interest-to-action gap: interest exceeds current studying by about {round(interest_pct - studying_pct, 1)} percentage points."
        )
    elif studying_pct >= interest_pct:
        sentences.append(
            "Current studying is keeping pace with or exceeding stated interest, which suggests strong follow-through."
        )

    if career_pct < interest_pct:
        sentences.append(
            f"Career intention is below raw interest by roughly {round(interest_pct - career_pct, 1)} points, so some students remain interested without committing to the field yet."
        )

    if not field_counts.empty:
        top_field = field_counts.index[0]
        top_field_pct = round(
            field_counts.iloc[0] / field_counts.sum() * 100, 1)
        sentences.append(
            f"The most popular IT field is {top_field} ({top_field_pct}% of field selections).")

    if not barrier_counts.empty:
        top_barrier = barrier_counts.index[0]
        top_barrier_pct = round(
            barrier_counts.iloc[0] / barrier_counts.sum() * 100, 1)
        sentences.append(
            f"The most common barrier is {top_barrier} ({top_barrier_pct}% of barrier selections).")

    if not grade_to_interest.empty:
        strongest_grade = grade_to_interest.mean(axis=1).idxmax()
        sentences.append(
            f"{strongest_grade} shows the strongest overall interest profile in the grade-to-interest table.")

    if not interest_to_study.empty and interest_to_study.max().max() < 50:
        sentences.append(
            "A notable share of interested students are still not studying IT consistently, which is a key conversion gap."
        )

    if interest_yes == 0 and studying_yes == 0 and career_yes == 0:
        sentences.append(
            "The current filter set is hiding most positive responses, so the dashboard is now showing a colder segment of the survey.")

    return sentences


def prepare_model(
    frame: pd.DataFrame, cols: dict[str, str]
) -> tuple[Pipeline | None, float | None, pd.DataFrame, list[str], list | None, list | None]:
    feature_cols = [cols["interest"], cols["studying"], cols["grade"]]
    target_col = cols["career"]
    model_frame = frame[feature_cols + [target_col]].copy()
    model_frame = model_frame.dropna(subset=[target_col])

    if model_frame.empty or model_frame[feature_cols].dropna(how="all").shape[0] < 6:
        return None, None, model_frame, feature_cols, None, None

    X = model_frame[feature_cols]
    y = model_frame[target_col].astype(str)

    if y.nunique() < 2:
        return None, None, model_frame, feature_cols, None, None

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=4, min_samples_leaf=4, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy, model_frame, feature_cols, list(y_test), list(y_pred)


def render_metric_card(title: str, value: str, caption: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="--accent: {accent};">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_response_words(opinions: pd.Series) -> pd.Series:
    stop_words = {
        "რომ",
        "და",
        "არის",
        "იყო",
        "the",
        "for",
        "with",
        "this",
        "that",
        "from",
        "you",
        "are",
        "but",
        "not",
        "რაც",
        "ერთი",
        "მერე",
        "ძალიან",
    }

    words = Counter()
    for opinion in opinions.dropna().astype(str):
        for token in re.findall(r"[\wა-ჰ]+", opinion.lower()):
            token = token.strip()
            if len(token) < 4 or token in stop_words:
                continue
            words[token] += 1

    if not words:
        return pd.Series(dtype=int)

    return pd.Series(dict(words.most_common(12)))


def load_and_filter_data(uploaded_file_bytes: bytes | None) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame, list[str], list[str]]:
    raw_df = load_data(uploaded_file_bytes)
    cols = resolve_columns(raw_df)

    work_df = raw_df.copy()
    for column in work_df.columns:
        if work_df[column].dtype == object:
            work_df[column] = work_df[column].replace({"": pd.NA})

    grade_values = sorted({class_to_str(value)
                          for value in work_df[cols["grade"]].dropna().unique()})
    gender_values = sorted({class_to_str(value)
                           for value in work_df[cols["gender"]].dropna().unique()})

    return raw_df, cols, work_df, grade_values, gender_values


# ─── UI render helpers ────────────────────────────────────────────────────────

def render_sidebar(
    work_df: pd.DataFrame,
    cols: dict[str, str],
    grade_values: list[str],
    gender_values: list[str],
    uploaded_file: object,
) -> tuple[list[str], list[str]]:
    """Render sidebar UI below the file uploader and return (selected_grades, selected_genders)."""

    # ── Dataset source indicator ──────────────────────────────────────────────
    if uploaded_file is not None:
        st.sidebar.markdown(
            '<span class="source-badge source-uploaded">⬆ Uploaded CSV</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(f"File: **{uploaded_file.name}**")
    else:
        st.sidebar.markdown(
            '<span class="source-badge source-sample">🔗 Sample dataset</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Using the linked Google Sheets sample.")

    st.sidebar.markdown("---")

    # ── Session-state keys: reset when data source changes ───────────────────
    source_key = uploaded_file.name if uploaded_file is not None else "__sample__"
    if st.session_state.get("_data_source_key") != source_key:
        st.session_state["_data_source_key"] = source_key
        st.session_state.pop("grade_filter", None)
        st.session_state.pop("gender_filter", None)

    # ── Filter controls ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")

    selected_grades: list[str] = st.sidebar.multiselect(
        "Grade level",
        grade_values,
        default=grade_values,
        key="grade_filter",
    )
    selected_genders: list[str] = st.sidebar.multiselect(
        "Gender",
        gender_values,
        default=gender_values,
        key="gender_filter",
    )

    filters_active = (
        set(selected_grades) != set(grade_values)
        or set(selected_genders) != set(gender_values)
    )
    if filters_active:
        if st.sidebar.button("↺ Reset filters", use_container_width=True):
            st.session_state["grade_filter"] = grade_values
            st.session_state["gender_filter"] = gender_values
            st.rerun()

    st.sidebar.markdown("---")

    # ── Dataset snapshot ──────────────────────────────────────────────────────
    st.sidebar.markdown("### Dataset snapshot")
    filtered_count = len(work_df)
    if selected_grades:
        filtered_count = len(
            work_df[work_df[cols["grade"]].map(class_to_str).isin(selected_grades)]
        )
    st.sidebar.markdown(
        f"""
        <div class="metric-card" style="--accent: #2563eb; min-height: auto; margin-bottom: 0.8rem;">
            <div class="metric-label">Rows after filters</div>
            <div class="metric-value" style="font-size: 1.7rem;">{filtered_count}</div>
            <div class="metric-caption">{len(work_df)} rows loaded from the source dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "Detected columns are mapped automatically from Georgian or English headers.")

    st.sidebar.markdown("---")

    # ── Export expander ───────────────────────────────────────────────────────
    with st.sidebar.expander("📥 Export", expanded=False):
        st.caption("Download the currently filtered dataset as a CSV file.")
        # Compute filtered_df here for the download (grades + genders)
        dl_df = work_df.copy()
        if selected_grades:
            dl_df = dl_df[dl_df[cols["grade"]].map(class_to_str).isin(selected_grades)]
        if selected_genders:
            dl_df = dl_df[dl_df[cols["gender"]].map(class_to_str).isin(selected_genders)]
        st.download_button(
            "Download filtered CSV",
            dl_df.to_csv(index=False).encode("utf-8"),
            file_name="it_career_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    return selected_grades, selected_genders


def render_active_filters_summary(
    selected_grades: list[str],
    grade_values: list[str],
    selected_genders: list[str],
    gender_values: list[str],
) -> None:
    """Show active-filter chips near the top of the main content area."""
    grade_filtered = set(selected_grades) != set(grade_values)
    gender_filtered = set(selected_genders) != set(gender_values)

    if not grade_filtered and not gender_filtered:
        return  # nothing to show when all defaults are active

    chips_html = '<div class="chips-row">'
    chips_html += '<span class="chips-note">Active filters:</span>'

    if grade_filtered:
        for g in sorted(selected_grades):
            chips_html += f'<span class="chip"><span class="chip-label">Grade</span>{g}</span>'

    if gender_filtered:
        for g in sorted(selected_genders):
            chips_html += f'<span class="chip"><span class="chip-label">Gender</span>{g}</span>'

    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)


def render_data_quality(
    filtered_df: pd.DataFrame, raw_df: pd.DataFrame, cols: dict[str, str]
) -> None:
    """Show a compact data-quality / coverage block below the KPI cards."""
    req_cols = {k: cols[k] for k in REQUIRED_COLUMNS if k in cols}

    items_html = ""
    for key, col in req_cols.items():
        total = len(filtered_df)
        missing = int(filtered_df[col].isna().sum())
        pct_missing = round(missing / total * 100, 1) if total else 0
        css_cls = "dq-good" if pct_missing == 0 else ("dq-warn" if pct_missing < 15 else "dq-bad")
        label = key.capitalize()
        items_html += (
            f'<div class="dq-item">'
            f'<div class="dq-item-label">{label} missing</div>'
            f'<div class="dq-item-val {css_cls}">{missing} ({pct_missing}%)</div>'
            f"</div>"
        )

    unique_grades = filtered_df[cols["grade"]].dropna().nunique()
    unique_genders = filtered_df[cols["gender"]].dropna().nunique()
    items_html += (
        f'<div class="dq-item">'
        f'<div class="dq-item-label">Unique grades</div>'
        f'<div class="dq-item-val dq-good">{unique_grades}</div>'
        f"</div>"
        f'<div class="dq-item">'
        f'<div class="dq-item-label">Unique genders</div>'
        f'<div class="dq-item-val dq-good">{unique_genders}</div>'
        f"</div>"
    )

    st.markdown(
        f'<div class="dq-block">'
        f'<div class="dq-title">📋 Data quality &amp; coverage</div>'
        f'<div class="dq-grid">{items_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_insights_section(insights: list[str]) -> None:
    """Show 1–2 key takeaways prominently, with the full list under an expander."""
    takeaways = insights[:2]
    rest = insights[2:]

    # Key takeaway cards
    cards_html = '<div class="takeaway-wrap">'
    for i, text in enumerate(takeaways):
        cards_html += (
            f'<div class="takeaway-card">'
            f'<div class="takeaway-label">💡 Key takeaway {i + 1}</div>'
            f'<div class="takeaway-text">{text}</div>'
            f"</div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Full list in expander
    if rest:
        with st.expander("View all insights", expanded=False):
            st.markdown(
                '<ul class="insight-list">' +
                "".join(f"<li>{s}</li>" for s in insights) +
                "</ul>",
                unsafe_allow_html=True,
            )


def render_overview(
    filtered_df: pd.DataFrame,
    cols: dict[str, str],
    interest_yes_pct: float,
) -> None:
    """Render the Overview charts tab."""
    section_heading(
        "Overview charts",
        "A clean, high-level view of what students selected, preferred, and struggled with.",
    )

    top_row_left, top_row_right = st.columns(2)
    with top_row_left:
        st.plotly_chart(
            make_pie(filtered_df[cols["interest"]], "Interest in IT",
                     ["#2563eb", "#7c3aed", "#14b8a6", "#f59e0b"]),
            use_container_width=True,
        )
        st.caption(f"{interest_yes_pct}% of visible responses indicate interest in IT.")
    with top_row_right:
        st.plotly_chart(
            make_pie(filtered_df[cols["gender"]], "Gender distribution",
                     ["#7c3aed", "#2563eb", "#14b8a6", "#f59e0b"]),
            use_container_width=True,
        )
        st.caption("This acts as the demographic anchor for the rest of the dashboard.")

    chart_cols = st.columns(3)
    field_series = (
        split_multiselect(filtered_df[cols["field"]]) if "field" in cols else pd.Series(dtype=int)
    )
    motivation_series = (
        split_multiselect(filtered_df[cols["motivation"]])
        if "motivation" in cols
        else pd.Series(dtype=int)
    )
    barrier_series = (
        split_multiselect(filtered_df[cols["barriers"]])
        if "barriers" in cols
        else pd.Series(dtype=int)
    )

    with chart_cols[0]:
        st.plotly_chart(
            make_horizontal_bar(field_series, "Most preferred IT fields", 1),
            use_container_width=True,
        )
        st.caption("This section shows where curiosity is concentrated.")
    with chart_cols[1]:
        st.plotly_chart(
            make_horizontal_bar(motivation_series, "Motivations for choosing IT", 2),
            use_container_width=True,
        )
        st.caption("Motivation is the strongest signal behind future action.")
    with chart_cols[2]:
        st.plotly_chart(
            make_horizontal_bar(barrier_series, "Barriers to learning IT", 3),
            use_container_width=True,
        )
        st.caption("The main friction points that slow down conversion into active study.")

    if "learning" in cols and cols["learning"] in filtered_df:
        st.plotly_chart(
            make_horizontal_bar(
                split_multiselect(filtered_df[cols["learning"]]),
                "Preferred learning methods",
                4,
            ),
            use_container_width=True,
        )
        st.caption(
            "Preferred learning methods help shape the next intervention or club activity."
        )


def render_relationships(filtered_df: pd.DataFrame, cols: dict[str, str]) -> None:
    """Render the Relationships tab (heatmaps + tables)."""
    section_heading(
        "Relationships",
        "This is where the dashboard moves from counts to structure and compares groups.",
    )

    rel_col1, rel_col2 = st.columns(2)
    study_interest_table = pct_table(filtered_df, cols["studying"], cols["interest"])
    career_interest_table = pct_table(filtered_df, cols["career"], cols["interest"])
    grade_interest_table = pct_table(filtered_df, cols["grade"], cols["interest"])

    with rel_col1:
        st.plotly_chart(
            make_heatmap(study_interest_table, "Interest vs studying status (%)"),
            use_container_width=True,
        )
        st.dataframe(format_pct_table(study_interest_table), use_container_width=True)
    with rel_col2:
        st.plotly_chart(
            make_heatmap(career_interest_table, "Interest vs career choice (%)"),
            use_container_width=True,
        )
        st.dataframe(format_pct_table(career_interest_table), use_container_width=True)

    st.plotly_chart(
        make_heatmap(grade_interest_table, "Grade vs interest (%)"),
        use_container_width=True,
    )
    st.dataframe(format_pct_table(grade_interest_table), use_container_width=True)


def render_prediction_lab(
    filtered_df: pd.DataFrame,
    work_df: pd.DataFrame,
    cols: dict[str, str],
    grade_values: list[str],
) -> None:
    """Render the Prediction Lab tab."""
    section_heading(
        "Prediction lab",
        "A lightweight model that estimates career choice from a few core survey answers.",
    )

    # Model info / disclaimer
    st.markdown(
        '<div class="model-info">'
        "<strong>How this works:</strong> A Decision Tree classifier is trained on the "
        "currently filtered slice using <em>interest level</em>, <em>studying status</em>, "
        "and <em>grade</em> as features to predict <em>career choice</em>. "
        "Results are indicative — the model re-trains live on each filter change and "
        "requires at least 6 labelled rows with two distinct career-choice classes. "
        "A 75/25 train/test split is used; the reported accuracy reflects the hold-out set."
        "</div>",
        unsafe_allow_html=True,
    )

    model, accuracy, model_frame, feature_cols, y_test, y_pred = prepare_model(
        filtered_df, cols
    )

    if model is None or accuracy is None:
        st.info(
            "⚠️ There is not enough balanced data to train a reliable model with the current "
            "filter selection. Try widening your grade or gender filters."
        )
        return

    # KPI row
    score_cols = st.columns(3)
    with score_cols[0]:
        render_metric_card(
            "Model accuracy",
            f"{accuracy * 100:.1f}%",
            "Hold-out score from the current filtered slice.",
            COLOR_SEQUENCE[4],
        )
    with score_cols[1]:
        render_metric_card(
            "Training rows",
            f"{len(model_frame)}",
            "Rows kept after removing incomplete target values.",
            COLOR_SEQUENCE[5],
        )
    with score_cols[2]:
        render_metric_card(
            "Input features",
            f"{len(feature_cols)}",
            "Interest, studying status, and grade.",
            COLOR_SEQUENCE[6],
        )

    # Feature importance
    classifier = model.named_steps["classifier"]
    importances = pd.Series(
        classifier.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    st.plotly_chart(
        make_horizontal_bar(importances, "Feature importance", 1),
        use_container_width=True,
    )

    # Class distribution of target
    career_dist = model_frame[cols["career"]].value_counts()
    with st.expander("Target class distribution", expanded=False):
        st.caption(
            "Shows how many rows belong to each career-choice category in the training data. "
            "Very imbalanced distributions reduce prediction reliability."
        )
        st.plotly_chart(
            make_horizontal_bar(career_dist, "Career choice distribution", 3),
            use_container_width=True,
        )

    # Confusion matrix (if test predictions available)
    if y_test is not None and y_pred is not None:
        classes = sorted(set(y_test) | set(y_pred))
        if len(classes) >= 2:
            with st.expander("Confusion matrix (hold-out set)", expanded=False):
                st.caption(
                    "Rows = actual labels, columns = predicted labels. "
                    "Diagonal cells are correct predictions."
                )
                cm = confusion_matrix(y_test, y_pred, labels=classes)
                cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                cm_fig = make_heatmap(
                    cm_df.astype(float), "Confusion matrix (counts)"
                )
                st.plotly_chart(cm_fig, use_container_width=True)

    # Interactive prediction form
    st.markdown("#### Try a prediction")
    with st.form("career_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            interest_choice = st.selectbox(
                "Interest level",
                sorted(
                    {class_to_str(v) for v in work_df[cols["interest"]].dropna().unique()}
                ),
            )
        with c2:
            studying_choice = st.selectbox(
                "Currently studying",
                sorted(
                    {class_to_str(v) for v in work_df[cols["studying"]].dropna().unique()}
                ),
            )
        with c3:
            grade_choice = st.selectbox("Grade", grade_values)
        submitted = st.form_submit_button("🔮 Predict career choice", use_container_width=True)

    if submitted:
        input_frame = pd.DataFrame(
            [
                {
                    cols["interest"]: interest_choice,
                    cols["studying"]: studying_choice,
                    cols["grade"]: grade_choice,
                }
            ]
        )
        prediction = model.predict(input_frame)[0]
        probability: float | None = None
        if hasattr(model, "predict_proba"):
            classes_list = list(classifier.classes_)
            if prediction in classes_list:
                probability = float(
                    model.predict_proba(input_frame)[0][classes_list.index(prediction)]
                )

        conf_str = (
            f'<div class="pred-result-conf">{probability * 100:.1f}% model confidence</div>'
            if probability is not None
            else ""
        )
        st.markdown(
            f'<div class="pred-result">'
            f'<div class="pred-result-label">Predicted career choice</div>'
            f'<div class="pred-result-value">{prediction}</div>'
            f"{conf_str}"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_open_responses(filtered_df: pd.DataFrame, cols: dict[str, str]) -> None:
    """Render the Open Responses tab."""
    section_heading(
        "Open responses",
        "Short text answers are often the most useful place to spot recurring themes.",
    )

    if "opinion" in cols and cols["opinion"] in filtered_df:
        opinions = filtered_df[cols["opinion"]].dropna().astype(str)
        if opinions.empty:
            st.info("No open-ended responses are available.")
        else:
            words = top_response_words(opinions)
            if not words.empty:
                st.plotly_chart(
                    make_horizontal_bar(words, "Common words in open responses", 2),
                    use_container_width=True,
                )
                st.caption(
                    "Only words with four or more characters are counted to keep the chart readable."
                )
            st.dataframe(
                opinions.head(50).to_frame(name="Open response"),
                use_container_width=True,
            )
    else:
        st.info("The uploaded dataset does not include an open-response column.")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    # ── Sidebar: file uploader (must be first widget) ─────────────────────────
    # We render a minimal uploader here to capture the file, then pass it to
    # render_sidebar() which builds the full sidebar UI.
    st.sidebar.markdown("### Data controls")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"],
                                              label_visibility="visible")

    uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

    try:
        raw_df, cols, work_df, grade_values, gender_values = load_and_filter_data(
            uploaded_bytes)
    except Exception as exc:
        st.error(f"Failed to load the survey data: {exc}")
        st.stop()

    # ── Sidebar (source indicator, filters, snapshot, export) ─────────────────
    selected_grades, selected_genders = render_sidebar(
        work_df, cols, grade_values, gender_values, uploaded_file
    )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered_df = work_df.copy()
    if selected_grades:
        filtered_df = filtered_df[
            filtered_df[cols["grade"]].map(class_to_str).isin(selected_grades)
        ]
    if selected_genders:
        filtered_df = filtered_df[
            filtered_df[cols["gender"]].map(class_to_str).isin(selected_genders)
        ]

    if filtered_df.empty:
        st.warning("No responses match the active filters.")
        st.stop()

    # ── Derived KPI values ────────────────────────────────────────────────────
    total_responses = len(filtered_df)
    interest_yes_count, interest_yes_pct = percent(
        filtered_df[cols["interest"]], POSITIVE_MARKERS["yes"])
    studying_yes_count, studying_yes_pct = percent(
        filtered_df[cols["studying"]], POSITIVE_MARKERS["yes"])
    career_yes_count, career_yes_pct = percent(
        filtered_df[cols["career"]], POSITIVE_MARKERS["yes"])

    insights = build_insights(filtered_df, cols)
    hero_summary = insights[0]
    hero_support = insights[1:3]

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">Survey intelligence dashboard</div>
                    <h1 class="hero-title">{APP_TITLE}</h1>
                    <div class="hero-copy">{hero_summary}</div>
                    <div class="mini-note">
                        {"<br>".join(f"• {line}" for line in hero_support) if hero_support else "The current filter set gives you a clean view of the survey segment you care about most."}
                    </div>
                </div>
                <div class="hero-panel">
                    <div class="hero-stat"><strong>{total_responses}</strong><span>Filtered responses in view</span></div>
                    <div class="hero-stat"><strong>{interest_yes_pct}%</strong><span>Interested in IT</span></div>
                    <div class="hero-stat"><strong>{studying_yes_pct}%</strong><span>Currently studying</span></div>
                    <div class="hero-stat"><strong>{career_yes_pct}%</strong><span>Planning IT career</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Active-filter chips ───────────────────────────────────────────────────
    render_active_filters_summary(
        selected_grades, grade_values, selected_genders, gender_values
    )

    # ── KPI metric cards ──────────────────────────────────────────────────────
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Total responses", f"{total_responses}",
                           "Active dataset rows after the selected filters.", COLOR_SEQUENCE[0])
    with metric_cols[1]:
        render_metric_card("Interested in IT", f"{interest_yes_pct}%",
                           f"{interest_yes_count} positive responses detected.", COLOR_SEQUENCE[1])
    with metric_cols[2]:
        render_metric_card("Currently studying", f"{studying_yes_pct}%",
                           f"{studying_yes_count} positive responses detected.", COLOR_SEQUENCE[2])
    with metric_cols[3]:
        render_metric_card("Planning IT career", f"{career_yes_pct}%",
                           f"{career_yes_count} positive responses detected.", COLOR_SEQUENCE[3])

    # ── Data quality block ────────────────────────────────────────────────────
    render_data_quality(filtered_df, raw_df, cols)

    # ── Key takeaways + insights expander ─────────────────────────────────────
    render_insights_section(insights)

    # ── Analysis tabs ─────────────────────────────────────────────────────────
    tab_overview, tab_relationships, tab_ml, tab_text = st.tabs(
        ["📊 Overview", "🔗 Relationships", "🤖 Prediction Lab", "💬 Open responses"]
    )

    with tab_overview:
        render_overview(filtered_df, cols, interest_yes_pct)

    with tab_relationships:
        render_relationships(filtered_df, cols)

    with tab_ml:
        render_prediction_lab(filtered_df, work_df, cols, grade_values)

    with tab_text:
        render_open_responses(filtered_df, cols)


if __name__ == "__main__":
    main()

