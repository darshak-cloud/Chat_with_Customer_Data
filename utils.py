"""Utility helpers for the Chat with Customer Data Streamlit app.

The app is intentionally data-first: Excel is parsed with Pandas, queries are
executed against DataFrames, and Gemini is used only for structured intent
hints and concise wording.
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import find_dotenv, load_dotenv


MONEY_UNITS = {
    "k": 1_000,
    "thousand": 1_000,
    "l": 100_000,
    "lac": 100_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "cr": 10_000_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
}


@dataclass(frozen=True)
class GeminiConfig:
    """Runtime configuration for Gemini-compatible summarization."""

    api_key: str | None
    model_name: str = "gemini-2.5-flash"
    enabled: bool = False


def load_gemini_config() -> GeminiConfig:
    """Load Gemini settings from .env/environment variables."""

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=True)
    else:
        sample_env_path = Path.cwd() / "sample.env"
        if sample_env_path.exists():
            load_dotenv(sample_env_path, override=True)
        else:
            load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        try:
            import streamlit as st

            project_secrets = Path.cwd() / ".streamlit" / "secrets.toml"
            user_secrets = Path.home() / ".streamlit" / "secrets.toml"
            if project_secrets.exists() or user_secrets.exists():
                api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
                model_name = st.secrets.get("GEMINI_MODEL", model_name)
        except Exception:
            pass

    enabled = bool(api_key)
    return GeminiConfig(api_key=api_key, model_name=model_name, enabled=enabled)


class GeminiHelper:
    """Small wrapper that keeps all Gemini calls optional and non-critical."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or load_gemini_config()
        self._client: Any | None = None
        self.last_error: str | None = None

        if not self.config.enabled:
            self.last_error = "GEMINI_API_KEY is not set."
            return

        try:
            from google import genai

            # Current Gemini docs recommend the Google Gen AI SDK. The client
            # reads GEMINI_API_KEY from the environment; passing it explicitly
            # also supports environments where automatic loading is unavailable.
            self._client = genai.Client(api_key=self.config.api_key)
        except Exception as exc:
            self._client = None
            self.last_error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str) -> str | None:
        """Return generated text, or None if Gemini is unavailable/fails."""

        if not self.available:
            return None
        try:
            response = self._client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
            )
            return (getattr(response, "text", "") or "").strip() or None
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def generate_json(self, prompt: str) -> dict[str, Any] | None:
        """Return a JSON object from Gemini, or None when parsing fails."""

        text = self.generate(prompt)
        if not text:
            return None

        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None


def normalize_column_name(name: str) -> str:
    """Normalize a column name for reliable matching."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def normalize_text(value: Any) -> str:
    """Normalize arbitrary values for case-insensitive search."""

    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_money_value(text: str) -> float | None:
    """Parse amounts like '90 lakhs', '70L', '1Cr', or '9000000' into INR."""

    cleaned = text.lower().replace(",", "").replace("rs.", "").replace("inr", "")
    money_pattern = re.compile(
        r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>crores?|cr|lakhs?|lacs?|lac|l|k|thousand)?"
    )
    match = money_pattern.search(cleaned)
    if not match:
        return None

    number = float(match.group("number"))
    unit = match.group("unit") or ""
    multiplier = MONEY_UNITS.get(unit, 1)
    return number * multiplier


def format_inr(value: Any) -> str:
    """Format a numeric INR value in a compact Indian real-estate style."""

    if value is None or pd.isna(value):
        return "N/A"
    amount = float(value)
    if amount >= 10_000_000:
        return f"INR {amount / 10_000_000:.2f} Cr"
    if amount >= 100_000:
        return f"INR {amount / 100_000:.2f} L"
    return f"INR {amount:,.0f}"


def load_excel_file(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded Excel file and lightly clean empty rows/columns."""

    df = pd.read_excel(uploaded_file)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric while tolerating commas and text noise."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(cleaned, errors="coerce")


def safe_preview(df: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    """Return a small display-safe preview."""

    return df.head(rows).copy()


def dataframe_to_records(df: pd.DataFrame, max_rows: int = 50) -> list[dict[str, Any]]:
    """Convert a DataFrame to serializable row records."""

    if df is None or df.empty:
        return []
    display_df = df.head(max_rows).copy()
    return display_df.where(pd.notna(display_df), None).to_dict(orient="records")


def summarize_columns(columns: Iterable[str]) -> str:
    """Human-readable schema summary for prompts and UI."""

    return ", ".join(str(col) for col in columns)
