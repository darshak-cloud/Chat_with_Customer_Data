"""Deterministic Pandas query engine for real-estate customer lead data.

Architecture:
User Question -> Intent Extraction -> Column Mapping -> Pandas Query Execution
-> Result Validation -> Summary Generation -> Final Response

The LLM is intentionally optional. It can help phrase summaries, but answers
and records always come from Pandas operations over the uploaded Excel data.
"""

from __future__ import annotations

import calendar
from difflib import get_close_matches
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils import (
    GeminiHelper,
    coerce_numeric,
    dataframe_to_records,
    format_inr,
    normalize_column_name,
    normalize_text,
    parse_money_value,
)


@dataclass
class QueryResult:
    answer: str
    summary: str
    records: pd.DataFrame = field(default_factory=pd.DataFrame)
    chart_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, Any] = field(default_factory=dict)
    intent: str = "unknown"
    filters_applied: list[str] = field(default_factory=list)


class CustomerDataQueryEngine:
    """Rule-first query engine backed by Pandas."""

    ROLE_SYNONYMS = {
        "name": ["name", "customer", "client", "lead"],
        "budget": ["budget", "price", "amount", "cost", "investment"],
        "property_type": ["property_type", "property", "type", "configuration", "bhk", "villa"],
        "location": ["location", "area", "locality", "city", "from", "place"],
        "contact": ["contact", "phone", "mobile", "number"],
        "possession": ["possession", "expected_possession", "handover"],
        "status": ["status", "call_status", "last_call_status", "connected"],
        "last_connected": ["last_call_connected_time", "connected_time", "call_time"],
    }

    COUNT_WORDS = ("count", "how many", "number of", "total")
    AVG_WORDS = ("average", "avg", "mean")
    MINMAX_WORDS = ("minimum", "maximum", "min", "max", "range")
    LIST_WORDS = ("list", "show", "give", "which", "display")
    TOP_WORDS = ("top", "highest", "most", "popular")
    SUMMARY_WORDS = ("summary", "summarize", "overview")
    MONTH_LOOKUP = {
        name.lower(): index
        for index, name in enumerate(calendar.month_name)
        if name
    } | {
        name.lower(): index
        for index, name in enumerate(calendar.month_abbr)
        if name
    } | {"sept": 9}

    def __init__(self, df: pd.DataFrame, gemini: GeminiHelper | None = None) -> None:
        if df is None or df.empty:
            raise ValueError("Uploaded Excel file does not contain any rows.")

        self.original_df = df.copy()
        self.df = df.copy()
        self.normalized_columns = {col: normalize_column_name(col) for col in self.df.columns}
        self.roles = self._detect_schema_roles()
        self.gemini = gemini or GeminiHelper()
        self._prepare_numeric_columns()
        self.schema_metadata = self._build_schema_metadata()

    def _detect_schema_roles(self) -> dict[str, str | None]:
        roles: dict[str, str | None] = {role: None for role in self.ROLE_SYNONYMS}
        for role, synonyms in self.ROLE_SYNONYMS.items():
            for column, normalized in self.normalized_columns.items():
                if normalized in synonyms or any(token in normalized for token in synonyms):
                    roles[role] = column
                    break
        return roles

    def _prepare_numeric_columns(self) -> None:
        budget_col = self.roles.get("budget")
        if budget_col:
            self.df[budget_col] = coerce_numeric(self.df[budget_col])

    def _build_schema_metadata(self) -> dict[str, dict[str, Any]]:
        """Create semantic metadata for every uploaded Excel column."""

        metadata: dict[str, dict[str, Any]] = {}
        for column in self.df.columns:
            series = self.df[column]
            normalized = normalize_column_name(column)
            non_null = series.dropna()
            numeric_values = coerce_numeric(series).dropna()
            parsed_dates = self._parse_date_series(series)

            is_numeric = not numeric_values.empty and len(numeric_values) >= max(1, int(len(non_null) * 0.5))
            is_date_like = (
                not parsed_dates.dropna().empty
                and len(parsed_dates.dropna()) >= max(1, int(len(non_null) * 0.4))
            ) or any(token in normalized for token in ("date", "time", "possession", "handover"))

            role = next((role_name for role_name, role_column in self.roles.items() if role_column == column), None)
            metadata[column] = {
                "normalized": normalized,
                "words": set(normalized.split("_")),
                "role": role,
                "is_numeric": is_numeric,
                "is_date_like": is_date_like,
                "samples": [str(value) for value in non_null.astype(str).drop_duplicates().head(25)],
            }
        return metadata

    def _parse_date_series(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if parsed.notna().any():
            return parsed

        month_year = series.astype(str).str.extract(
            r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<year>20\d{2})",
            flags=re.IGNORECASE,
        )
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        for index, row in month_year.dropna().iterrows():
            month = self.MONTH_LOOKUP.get(str(row["month"]).lower())
            year = int(row["year"])
            if month:
                result.loc[index] = pd.Timestamp(year=year, month=month, day=1)
        return result

    def _answer_with_dynamic_excel_planner(self, question: str, q: str) -> QueryResult | None:
        """Generic schema-aware planner for arbitrary Excel columns and values."""

        if self._is_high_intent_summary_question(q):
            working_df = self.df.copy()
            working_df, high_intent_filter = self._apply_high_intent_filter(working_df)
            filters = [high_intent_filter]
            result = self._build_high_intent_summary_result(working_df, filters)
            result.answer = self._generate_final_answer(question, result)
            result.summary = self._summarize(question, result)
            return result

        plan = self._build_dynamic_plan(question, q)
        if not plan:
            return None

        working_df = self.df.copy()
        filters: list[str] = []

        for filter_spec in plan["filters"]:
            working_df = self._apply_dynamic_filter(working_df, filter_spec, filters)

        select_columns = [column for column in plan["select_columns"] if column in self.df.columns]
        if not select_columns:
            select_columns = list(self.df.columns)

        intent = plan["intent"]
        group_by = plan.get("group_by")
        aggregation = plan.get("aggregation")
        limit = int(plan.get("limit") or self._extract_top_n(q))

        if group_by and group_by in working_df.columns and aggregation == "count":
            grouped_label = self._group_count_label(question, group_by)
            grouped = (
                working_df[group_by]
                .dropna()
                .astype(str)
                .value_counts()
                .rename_axis(group_by)
                .reset_index(name=grouped_label)
            )
            if intent == "top":
                grouped = grouped.head(limit)

            answer = f"{len(grouped)} grouped row(s) were generated."
            if intent == "top":
                answer = f"Top {len(grouped)} {group_by} value(s) by {grouped_label.lower()} are shown below."
            elif intent == "count":
                answer = f"Counts by {group_by} are shown below."

            return QueryResult(
                answer=answer,
                records=self._prepare_records_for_display(grouped),
                chart_data=grouped.set_index(group_by) if not grouped.empty else grouped,
                metrics={
                    "Matching Records": int(len(working_df)),
                    "Grouped Rows": int(len(grouped)),
                },
                intent=f"dynamic_{intent}",
                filters_applied=filters,
                summary=""
            )

        if intent == "count":
            count = int(len(working_df))
            display_columns = self._display_columns_for_count(select_columns, plan.get("metric_column"))
            return QueryResult(
                answer=f"{count} matching record(s) found.",
                records=self._prepare_records_for_display(working_df[display_columns]),
                metrics={"Matching Records": count},
                intent="dynamic_count",
                filters_applied=filters,
                summary=""
            )

        if intent in {"average", "min_max", "top"}:
            numeric_column = plan.get("metric_column") or self._first_numeric_column_from_columns(select_columns)
            if numeric_column and numeric_column in working_df.columns:
                values = coerce_numeric(working_df[numeric_column]).dropna()
                if intent == "average":
                    if values.empty:
                        answer = f"No numeric values were found in {numeric_column}."
                        metrics = {"Matching Records": int(len(working_df))}
                    else:
                        avg = values.mean()
                        formatted = self._format_column_value(numeric_column, avg)
                        answer = f"The average {numeric_column} is {formatted}."
                        metrics = {f"Average {numeric_column}": formatted, "Matching Records": int(len(working_df))}
                    return QueryResult(answer=answer, records=self._prepare_records_for_display(working_df[select_columns]), metrics=metrics, intent="dynamic_average", filters_applied=filters, summary="")

                if intent == "min_max":
                    if values.empty:
                        answer = f"No numeric values were found in {numeric_column}."
                        metrics = {"Matching Records": int(len(working_df))}
                    else:
                        min_value = self._format_column_value(numeric_column, values.min())
                        max_value = self._format_column_value(numeric_column, values.max())
                        answer = f"The minimum {numeric_column} is {min_value} and the maximum {numeric_column} is {max_value}."
                        metrics = {f"Minimum {numeric_column}": min_value, f"Maximum {numeric_column}": max_value}
                    return QueryResult(answer=answer, records=self._prepare_records_for_display(working_df[select_columns]), metrics=metrics, intent="dynamic_min_max", filters_applied=filters, summary="")

                if intent == "top":
                    top_n = self._extract_top_n(q)
                    sorted_df = working_df.sort_values(by=numeric_column, ascending=False, na_position="last").head(top_n)
                    answer = f"Top {len(sorted_df)} record(s) by {numeric_column} are shown below."
                    return QueryResult(
                        answer=answer,
                        records=self._prepare_records_for_display(sorted_df[select_columns]),
                        metrics={"Matching Records": int(len(working_df))},
                        intent="dynamic_top",
                        filters_applied=filters,
                        summary=""
                    )

        records = self._prepare_records_for_display(working_df[select_columns])
        answer = f"{len(working_df)} matching record(s) found."
        if records.empty:
            answer = self._build_zero_match_answer(filters)
        elif len(select_columns) <= 4:
            answer = self._build_compact_dynamic_answer(records, select_columns, len(working_df))
        return QueryResult(
            answer=answer,
            records=records,
            metrics={"Matching Records": int(len(working_df))},
            intent="dynamic_list",
            filters_applied=filters,
            summary=""
        )

    def _build_dynamic_plan(self, question: str, q: str) -> dict[str, Any] | None:
        llm_plan = self._build_gemini_dynamic_plan(question)
        select_columns = self._infer_select_columns(q)
        filters = self._infer_dynamic_filters(q)
        intent = self._infer_dynamic_intent(q)
        metric_column = self._infer_metric_column(q, select_columns)
        group_by = None
        aggregation = None
        limit = self._extract_top_n(q)

        if llm_plan:
            llm_select_columns = self._validate_columns(llm_plan.get("select_columns"))
            llm_filters = self._validate_dynamic_filters(llm_plan.get("filters"))
            llm_group_by = self._match_column(str(llm_plan.get("group_by", ""))) if llm_plan.get("group_by") else None
            llm_aggregation = llm_plan.get("aggregation") if llm_plan.get("aggregation") in {"count"} else None
            llm_limit = llm_plan.get("limit")
            select_columns = list(dict.fromkeys(llm_select_columns + select_columns))
            filters = llm_filters + filters
            intent = llm_plan.get("intent") or intent
            metric_column = llm_plan.get("metric_column") or metric_column
            group_by = llm_group_by or group_by
            aggregation = llm_aggregation or aggregation
            if isinstance(llm_limit, int) and llm_limit > 0:
                limit = min(llm_limit, 50)

        heuristic_group_by = self._infer_group_by_column(q, select_columns)
        if not group_by:
            group_by = heuristic_group_by
        if group_by and not aggregation:
            aggregation = "count"

        select_columns = self._validate_columns(select_columns)
        filters = self._validate_dynamic_filters(filters)
        metric_column = metric_column if metric_column in self.df.columns else None

        has_dynamic_signal = bool(select_columns or filters or metric_column or group_by)
        if not has_dynamic_signal:
            return None

        if not select_columns:
            select_columns = self._default_display_columns()
        elif self._asks_for_entity_rows(q) and "only" not in q:
            base_columns = self._default_display_columns()
            select_columns = list(dict.fromkeys(base_columns + select_columns))
        if metric_column and metric_column not in select_columns:
            select_columns.append(metric_column)
        if group_by and group_by not in select_columns and "only" not in q:
            select_columns.insert(0, group_by)

        return {
            "intent": intent,
            "select_columns": select_columns,
            "filters": filters,
            "metric_column": metric_column,
            "group_by": group_by,
            "aggregation": aggregation,
            "limit": limit,
        }

    def _build_gemini_dynamic_plan(self, question: str) -> dict[str, Any] | None:
        if not self.gemini.available:
            return None

        schema = [
            {
                "column": column,
                "role": meta["role"],
                "is_numeric": meta["is_numeric"],
                "is_date_like": meta["is_date_like"],
                "samples": meta["samples"][:10],
            }
            for column, meta in self.schema_metadata.items()
        ]
        prompt = f"""
Create a JSON query plan for answering a question about an Excel dataframe.
Use only columns from the schema. Do not answer the question.

Allowed intents: list, count, average, min_max, top.
Filters support:
- exact: {{"column": "...", "operator": "equals", "value": "..."}}
- contains: {{"column": "...", "operator": "contains", "value": "..."}}
- month_year: {{"column": "...", "operator": "month_year", "month": 9, "year": 2027}}
- not_contains: {{"column": "...", "operator": "not_contains", "value": "..."}}
- in_values: {{"column": "...", "operator": "in_values", "values": ["...", "..."]}}

Return only valid JSON:
{{
  "intent": "list",
  "select_columns": ["..."],
  "group_by": null,
  "aggregation": null,
  "limit": null,
  "metric_column": null,
  "filters": []
}}

Schema: {schema}
Question: {question}
Important:
- Keep all explicitly requested output columns from the user question, such as name, budget, location, contact, or status.
- If the user says "only", return only those requested columns in select_columns.
- If the user asks to show/list customers, leads, records, or rows, and does not say "only", return row-level columns that identify the customer, not just a single metric column.
- Map natural-language call-status questions to the real status values from the schema samples. For example, "not receive my call", "missed my call", "not reachable", or "wasn't able to reach them" should become a status filter, not an unfiltered list.
- For questions like "top 5 locations by customer interest", use:
  intent = "top"
  select_columns = ["Location"]
  group_by = "Location"
  aggregation = "count"
  limit = 5
  metric_column = null
  filters = []
- "customer interest" means count matching rows grouped by the requested column.
- For questions like "how many leads came from Mumbai", use:
  intent = "count"
  filters = [{{"column": "Location", "operator": "contains", "value": "Mumbai"}}] only if a location-like column exists and the question clearly asks for a place filter.
- If the user asks for top N/grouped results, prefer group_by + aggregation instead of returning raw rows.
"""
        plan = self.gemini.generate_json(prompt)
        return plan if isinstance(plan, dict) else None

    def _infer_group_by_column(self, q: str, select_columns: list[str]) -> str | None:
        if not any(word in q for word in self.TOP_WORDS):
            return None

        requested_columns = self._find_requested_columns(q)
        candidate_columns = requested_columns + select_columns
        for column in candidate_columns:
            if column in self.df.columns and not self.schema_metadata[column]["is_numeric"]:
                return column

        if "customer interest" in q and self.roles.get("location"):
            return self.roles["location"]
        return None

    def _group_count_label(self, question: str, group_by: str) -> str:
        normalized_question = normalize_text(question)
        if "customer interest" in normalized_question:
            return "Customer Interest Count"
        return f"{group_by} Count"

    def _asks_for_entity_rows(self, q: str) -> bool:
        row_terms = (
            "customer",
            "customers",
            "client",
            "clients",
            "lead",
            "leads",
            "record",
            "records",
            "row",
            "rows",
        )
        return any(re.search(rf"\b{re.escape(term)}\b", q) for term in row_terms)

    def _infer_dynamic_intent(self, q: str) -> str:
        if any(word in q for word in self.SUMMARY_WORDS):
            return "summary"
        if any(word in q for word in self.COUNT_WORDS):
            return "count"
        if any(word in q for word in self.AVG_WORDS):
            return "average"
        if any(word in q for word in self.MINMAX_WORDS):
            return "min_max"
        if any(word in q for word in self.TOP_WORDS):
            return "top"
        return "list"

    def _infer_select_columns(self, q: str) -> list[str]:
        requested = self._find_requested_columns(q)
        if any(word in q for word in ("name", "names")) and self.roles.get("name"):
            requested.insert(0, self.roles["name"])
        requested = list(dict.fromkeys(requested))
        if "only" in q:
            return requested
        if requested:
            return requested
        return requested

    def _infer_metric_column(self, q: str, select_columns: list[str]) -> str | None:
        for column in self._find_requested_columns(q) + select_columns:
            if column in self.schema_metadata and self.schema_metadata[column]["is_numeric"]:
                return column
        return None

    def _infer_dynamic_filters(self, q: str) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []

        month_year = self._extract_month_year(q)
        if month_year:
            month, year = month_year
            date_column = self._best_date_column_for_question(q)
            if date_column:
                filters.append({"column": date_column, "operator": "month_year", "month": month, "year": year})

        for column, value in self._find_fuzzy_cell_value_matches(q):
            if month_year and self.schema_metadata[column]["is_date_like"]:
                continue
            filters.append({"column": column, "operator": "equals", "value": value})

        property_col = self.roles.get("property_type")
        if property_col and any(word in q for word in ("flat", "flats", "apartment", "apartments")):
            values = [normalize_text(value) for value in self.df[property_col].dropna().astype(str).unique()]
            if any("villa" in value for value in values):
                filters.append({"column": property_col, "operator": "not_contains", "value": "villa"})
            if any("bhk" in value for value in values):
                filters.append({"column": property_col, "operator": "contains", "value": "BHK"})

        filters.extend(self._infer_location_phrase_filters(q))
        filters.extend(self._infer_numeric_filters(q))
        filters.extend(self._infer_semantic_status_filters(q))

        return filters

    def _infer_semantic_status_filters(self, q: str) -> list[dict[str, Any]]:
        status_col = self.roles.get("status")
        if not status_col or status_col not in self.df.columns:
            return []

        normalized_q = normalize_text(q)
        negative_call_signals = (
            "not receive my call",
            "did not receive my call",
            "didn't receive my call",
            "missed my call",
            "not able to reach",
            "wasnt able to reach",
            "wasn't able to reach",
            "unable to reach",
            "not reachable",
            "could not reach",
            "couldn't reach",
            "no answer",
            "busy",
        )
        if not any(signal in normalized_q for signal in negative_call_signals):
            return []

        status_values = [str(value) for value in self.df[status_col].dropna().unique()]
        negative_tokens = {
            "busy",
            "no answer",
            "not connected",
            "not reachable",
            "unreachable",
            "switched off",
            "missed",
            "call back",
            "voicemail",
            "later",
            "did not answer",
            "not interested later",
        }
        matched_values = [
            value
            for value in status_values
            if any(token in normalize_text(value) for token in negative_tokens)
        ]
        if not matched_values:
            return []

        return [{"column": status_col, "operator": "in_values", "values": matched_values}]

    def _infer_location_phrase_filters(self, q: str) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        location_col = self.roles.get("location")
        if not location_col:
            return filters

        if self._find_cell_value_matches(q):
            return filters

        city_match = re.search(r"\b(?:from|in|near|at)\b\s+([a-z][a-z\s-]{1,30})", q)
        if not city_match:
            return filters

        requested = city_match.group(1).strip()
        stop_words = {
            "customers",
            "customer",
            "leads",
            "lead",
            "with",
            "budget",
            "looking",
            "for",
            "came",
            "come",
            "who",
            "that",
            "the",
        }
        requested = " ".join(word for word in requested.split() if word not in stop_words)
        if requested and not self._looks_like_month_phrase(requested):
            filters.append({"column": location_col, "operator": "contains", "value": requested})
        return filters

    def _infer_numeric_filters(self, q: str) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        numeric_column = self._infer_metric_column(q, self._find_requested_columns(q))
        if not numeric_column:
            budget_col = self.roles.get("budget")
            if budget_col and any(word in q for word in ("budget", "price", "amount", "investment")):
                numeric_column = budget_col
        if not numeric_column:
            return filters

        between_match = re.search(
            r"between\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)\s+and\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)",
            q,
        )
        if between_match:
            low = parse_money_value(between_match.group(1))
            high = parse_money_value(between_match.group(2))
            if low is not None and high is not None:
                low, high = sorted([low, high])
                filters.append({"column": numeric_column, "operator": "between", "low": low, "high": high})
            return filters

        comparator_patterns = [
            (r"(?:above|over|greater than|more than|higher than)\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)", ">"),
            (r"(?:below|under|less than|lower than)\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)", "<"),
            (r"(?:at least|minimum of)\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)", ">="),
            (r"(?:up to|upto|maximum of)\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)", "<="),
        ]
        for pattern, operator in comparator_patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            amount = parse_money_value(match.group(1))
            if amount is not None:
                filters.append({"column": numeric_column, "operator": operator, "value": amount})
            break

        return filters

    def _extract_month_year(self, q: str) -> tuple[int, int] | None:
        month_pattern = "|".join(sorted((re.escape(month) for month in self.MONTH_LOOKUP), key=len, reverse=True))
        match = re.search(rf"\b(?P<month>{month_pattern})\b\s*,?\s*(?P<year>20\d{{2}})\b", q)
        if not match:
            return None
        month = self.MONTH_LOOKUP.get(match.group("month").lower())
        year = int(match.group("year"))
        return (month, year) if month else None

    def _looks_like_month_phrase(self, text: str) -> bool:
        normalized = normalize_text(text)
        if not normalized:
            return False

        if re.search(r"\b20\d{2}\b", normalized):
            month_tokens = set(self.MONTH_LOOKUP.keys())
            if any(token in month_tokens for token in normalized.split()):
                return True

        return normalized in self.MONTH_LOOKUP

    def _best_date_column_for_question(self, q: str) -> str | None:
        date_columns = [column for column, meta in self.schema_metadata.items() if meta["is_date_like"]]
        if not date_columns:
            return None

        possession_words = ("possess", "possession", "handover", "move", "moving")
        if any(word in q for word in possession_words) and self.roles.get("possession") in date_columns:
            return self.roles["possession"]

        for column in date_columns:
            normalized = self.schema_metadata[column]["normalized"].replace("_", " ")
            if normalized in q:
                return column
        return date_columns[0]

    def _find_fuzzy_cell_value_matches(self, q: str) -> list[tuple[str, str]]:
        exact_matches = self._find_cell_value_matches(q)
        if exact_matches:
            return exact_matches

        matches: list[tuple[str, str]] = []
        for column in self.df.columns:
            values = [str(value) for value in self.df[column].dropna().astype(str).unique()]
            normalized_values = {normalize_text(value): value for value in values if len(normalize_text(value)) >= 3}
            for normalized_value, original_value in normalized_values.items():
                if self._value_tokens_present(normalized_value, q):
                    matches.append((column, original_value))
                    break

            if column in [match[0] for match in matches]:
                continue

            choices = list(normalized_values.keys())
            close = get_close_matches(q, choices, n=1, cutoff=0.82)
            if close:
                matches.append((column, normalized_values[close[0]]))

        matches.sort(key=lambda item: len(normalize_text(item[1])), reverse=True)
        deduped: list[tuple[str, str]] = []
        seen_columns: set[str] = set()
        for column, value in matches:
            if column not in seen_columns:
                deduped.append((column, value))
                seen_columns.add(column)
        return deduped

    def _value_tokens_present(self, normalized_value: str, q: str) -> bool:
        tokens = [token for token in normalized_value.split() if len(token) >= 3]
        return bool(tokens) and all(re.search(rf"\b{re.escape(token)}\b", q) for token in tokens)

    def _validate_columns(self, columns: Any) -> list[str]:
        if not isinstance(columns, list):
            return []
        valid: list[str] = []
        for column in columns:
            matched = self._match_column(str(column))
            if matched and matched not in valid:
                valid.append(matched)
        return valid

    def _validate_dynamic_filters(self, filters: Any) -> list[dict[str, Any]]:
        if not isinstance(filters, list):
            return []
        validated: list[dict[str, Any]] = []
        allowed = {"equals", "contains", "not_contains", "month_year", "between", ">", ">=", "<", "<=", "in_values"}
        for filter_spec in filters:
            if not isinstance(filter_spec, dict):
                continue
            column = self._match_column(str(filter_spec.get("column", "")))
            operator = filter_spec.get("operator")
            if not column or operator not in allowed:
                continue
            clean_filter = {"column": column, "operator": operator}
            if operator == "month_year":
                month = filter_spec.get("month")
                year = filter_spec.get("year")
                if not isinstance(month, int) or not isinstance(year, int):
                    continue
                clean_filter.update({"month": month, "year": year})
            elif operator == "between":
                low = filter_spec.get("low")
                high = filter_spec.get("high")
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                    continue
                clean_filter.update({"low": float(low), "high": float(high)})
            elif operator in {">", ">=", "<", "<="}:
                value = filter_spec.get("value")
                if not isinstance(value, (int, float)):
                    continue
                clean_filter["value"] = float(value)
            elif operator == "in_values":
                values = filter_spec.get("values")
                if not isinstance(values, list) or not values:
                    continue
                clean_values = [str(value) for value in values if value is not None]
                if not clean_values:
                    continue
                clean_filter["values"] = clean_values
            else:
                value = filter_spec.get("value")
                if value is None:
                    continue
                clean_filter["value"] = str(value)
            if clean_filter not in validated:
                validated.append(clean_filter)
        return validated

    def _match_column(self, column_hint: str) -> str | None:
        normalized_hint = normalize_column_name(column_hint)
        for column, normalized in self.normalized_columns.items():
            if normalized_hint == normalized:
                return column
        close = get_close_matches(normalized_hint, list(self.normalized_columns.values()), n=1, cutoff=0.72)
        if close:
            return next(column for column, normalized in self.normalized_columns.items() if normalized == close[0])
        return None

    def _apply_dynamic_filter(
        self, df: pd.DataFrame, filter_spec: dict[str, Any], filters_applied: list[str]
    ) -> pd.DataFrame:
        column = filter_spec["column"]
        operator = filter_spec["operator"]

        if operator == "month_year":
            parsed = self._parse_date_series(df[column])
            month = filter_spec["month"]
            year = filter_spec["year"]
            filters_applied.append(f"{column} month/year = {calendar.month_abbr[month]} {year}")
            return df[(parsed.dt.month == month) & (parsed.dt.year == year)]

        if operator == "between":
            low = filter_spec["low"]
            high = filter_spec["high"]
            numeric_series = coerce_numeric(df[column])
            filters_applied.append(f"{column} between {self._format_column_value(column, low)} and {self._format_column_value(column, high)}")
            return df[numeric_series.between(low, high, inclusive="both")]

        if operator in {">", ">=", "<", "<="}:
            numeric_series = coerce_numeric(df[column])
            value = float(filter_spec["value"])
            filters_applied.append(f"{column} {operator} {self._format_column_value(column, value)}")
            if operator == ">":
                return df[numeric_series > value]
            if operator == ">=":
                return df[numeric_series >= value]
            if operator == "<":
                return df[numeric_series < value]
            return df[numeric_series <= value]

        if operator == "in_values":
            raw_values = [str(item) for item in filter_spec.get("values", [])]
            values = [item.lower() for item in raw_values]
            filters_applied.append(f"{column} in {', '.join(raw_values)}")
            return df[df[column].astype(str).str.lower().isin(values)]

        value = str(filter_spec["value"])
        if operator == "equals":
            matched_value = self._closest_existing_value(column, value)
            filters_applied.append(f"{column} = {matched_value}")
            return df[df[column].astype(str).str.lower() == str(matched_value).lower()]
        if operator == "contains":
            filters_applied.append(f"{column} contains '{value}'")
            return df[df[column].astype(str).str.contains(value, case=False, na=False)]
        if operator == "not_contains":
            filters_applied.append(f"{column} does not contain '{value}'")
            return df[~df[column].astype(str).str.contains(value, case=False, na=False)]
        return df

    def _closest_existing_value(self, column: str, value: str) -> str:
        values = [str(item) for item in self.df[column].dropna().astype(str).unique()]
        normalized_map = {normalize_text(item): item for item in values}
        normalized_value = normalize_text(value)
        if normalized_value in normalized_map:
            return normalized_map[normalized_value]
        close = get_close_matches(normalized_value, list(normalized_map.keys()), n=1, cutoff=0.72)
        return normalized_map[close[0]] if close else value

    def _default_display_columns(self) -> list[str]:
        preferred_roles = ("name", "budget", "property_type", "location", "contact", "possession", "status")
        columns = [self.roles[role] for role in preferred_roles if self.roles.get(role)]
        return list(dict.fromkeys(columns)) or list(self.df.columns)

    def _display_columns_for_count(self, select_columns: list[str], metric_column: str | None) -> list[str]:
        columns: list[str] = []
        for role in ("name", "contact", "location", "property_type", "status"):
            column = self.roles.get(role)
            if column and column not in columns:
                columns.append(column)
        for column in select_columns:
            if column and column not in columns:
                columns.append(column)
        if metric_column and metric_column not in columns:
            columns.append(metric_column)
        return [column for column in columns if column in self.df.columns] or self._default_display_columns()

    def _first_numeric_column_from_columns(self, columns: list[str]) -> str | None:
        for column in columns:
            if column in self.schema_metadata and self.schema_metadata[column]["is_numeric"]:
                return column
        return None

    def _build_compact_dynamic_answer(self, records: pd.DataFrame, columns: list[str], total_rows: int) -> str:
        if records.empty:
            return "No matching records were found in the uploaded Excel data."
        parts = []
        for _, row in records.head(5).iterrows():
            values = [f"{column}: {self._format_column_value(column, row[column])}" for column in columns]
            parts.append("; ".join(values))
        suffix = f" Showing first {min(len(records), 5)}." if total_rows > 5 else ""
        return f"{total_rows} matching record(s) found. " + " | ".join(parts) + suffix

    def _build_zero_match_answer(self, filters: list[str]) -> str:
        if not filters:
            return "No matching records were found in the uploaded Excel data."
        return "No matching records were found for the selected conditions: " + "; ".join(filters) + "."

    def answer_question(self, question: str) -> QueryResult:
        question = (question or "").strip()
        if not question:
            return QueryResult(
                answer="Please ask a question about the uploaded customer data.",
                summary="No question was provided.",
            )

        q = normalize_text(question)
        dynamic_result = self._answer_with_dynamic_excel_planner(question, q)
        if dynamic_result is not None:
            dynamic_result.answer = self._generate_final_answer(question, dynamic_result)
            dynamic_result.summary = self._summarize(question, dynamic_result)
            return dynamic_result

        llm_hints = self._extract_llm_hints(question)
        rule_intent = self._extract_intent(q)
        if rule_intent in {
            "generic_attribute",
            "generic_average",
            "generic_min_max",
            "generic_top",
            "customer_attribute",
            "top_budget_customer",
        }:
            intent = rule_intent
        else:
            intent = self._validated_llm_intent(llm_hints) or rule_intent
        working_df = self.df.copy()
        filters: list[str] = []

        working_df, filters = self._apply_filters(working_df, q, filters)
        working_df, filters = self._apply_validated_llm_filters(working_df, llm_hints, filters)

        if "high_intent" in q.replace("-", "_") or "high intent" in q:
            working_df, high_intent_filter = self._apply_high_intent_filter(working_df)
            filters.append(high_intent_filter)
            intent = "summary" if intent == "unknown" else intent

        result = self._execute(intent, q, working_df, filters)
        result.answer = self._generate_final_answer(question, result)
        result.summary = self._summarize(question, result)
        return result

    def _extract_llm_hints(self, question: str) -> dict[str, Any]:
        """Ask Gemini for query intent and filters without trusting it directly."""

        if not self.gemini.available:
            return {}

        property_col = self.roles.get("property_type")
        location_col = self.roles.get("location")
        status_col = self.roles.get("status")

        property_values = self._sample_unique_values(property_col)
        location_values = self._sample_unique_values(location_col)
        status_values = self._sample_unique_values(status_col)

        prompt = f"""
You translate user questions about an uploaded customer Excel file into JSON.
Do not answer the question. Do not calculate numbers.

Allowed intents: count, list, generic_attribute, generic_average, generic_min_max, generic_top, customer_attribute, average_budget, budget_min_max, top_budget_customer, top_locations, summary.
Available columns: {list(self.df.columns)}
Detected roles: {self.roles}
Known property values: {property_values}
Known location values: {location_values}
Known status values: {status_values}

Return only valid JSON with this shape:
{{
  "intent": "one allowed intent",
  "property_type": "exact value from known property values or null",
  "location": "exact value from known location values or null",
  "status": "exact value from known status values or null",
  "budget_operator": one of [">", ">=", "<", "<=", "between", null],
  "budget_value": "amount text like 90 lakhs or null",
  "budget_value_2": "second amount text for between queries or null",
  "top_n": integer or null
}}

Question: {question}
"""
        hints = self.gemini.generate_json(prompt)
        return hints or {}

    def _validated_llm_intent(self, hints: dict[str, Any]) -> str | None:
        allowed = {
            "count",
            "list",
            "generic_attribute",
            "generic_average",
            "generic_min_max",
            "generic_top",
            "customer_attribute",
            "average_budget",
            "budget_min_max",
            "top_budget_customer",
            "top_locations",
            "summary",
        }
        intent = normalize_text(hints.get("intent")) if hints else ""
        return intent if intent in allowed else None

    def _apply_validated_llm_filters(
        self, df: pd.DataFrame, hints: dict[str, Any], filters: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply only Gemini suggestions that validate against real DataFrame data."""

        if not hints:
            return df, filters

        df = self._apply_validated_value_filter(
            df=df,
            column=self.roles.get("property_type"),
            suggested_value=hints.get("property_type"),
            filters=filters,
            source_label="Gemini-validated property filter",
        )
        df = self._apply_validated_value_filter(
            df=df,
            column=self.roles.get("location"),
            suggested_value=hints.get("location"),
            filters=filters,
            source_label="Gemini-validated location filter",
        )
        df = self._apply_validated_value_filter(
            df=df,
            column=self.roles.get("status"),
            suggested_value=hints.get("status"),
            filters=filters,
            source_label="Gemini-validated status filter",
        )
        df = self._apply_validated_budget_hint(df, hints, filters)
        return df, filters

    def _apply_validated_value_filter(
        self,
        df: pd.DataFrame,
        column: str | None,
        suggested_value: Any,
        filters: list[str],
        source_label: str,
    ) -> pd.DataFrame:
        if not column or not suggested_value:
            return df

        actual_values = [str(value) for value in df[column].dropna().unique()]
        normalized_suggestion = normalize_text(suggested_value)
        matched = next(
            (value for value in actual_values if normalize_text(value) == normalized_suggestion),
            None,
        )
        if not matched:
            return df

        filter_text = f"{source_label}: {column} = {matched}"
        if filter_text not in filters:
            filters.append(filter_text)
        return df[df[column].astype(str).str.lower() == matched.lower()]

    def _apply_validated_budget_hint(
        self, df: pd.DataFrame, hints: dict[str, Any], filters: list[str]
    ) -> pd.DataFrame:
        budget_col = self.roles.get("budget")
        operator = hints.get("budget_operator")
        first_value = hints.get("budget_value")
        if not budget_col or not operator or not first_value:
            return df

        first_amount = parse_money_value(str(first_value))
        if first_amount is None:
            return df

        if operator == "between":
            second_amount = parse_money_value(str(hints.get("budget_value_2") or ""))
            if second_amount is None:
                return df
            low, high = sorted([first_amount, second_amount])
            filters.append(f"Gemini-validated budget filter: {budget_col} between {format_inr(low)} and {format_inr(high)}")
            return df[df[budget_col].between(low, high, inclusive="both")]

        if operator not in {">", ">=", "<", "<="}:
            return df

        filters.append(f"Gemini-validated budget filter: {budget_col} {operator} {format_inr(first_amount)}")
        if operator == ">":
            return df[df[budget_col] > first_amount]
        if operator == ">=":
            return df[df[budget_col] >= first_amount]
        if operator == "<":
            return df[df[budget_col] < first_amount]
        return df[df[budget_col] <= first_amount]

    def _sample_unique_values(self, column: str | None, limit: int = 30) -> list[str]:
        if not column or column not in self.df.columns:
            return []
        return [str(value) for value in self.df[column].dropna().unique()[:limit]]

    def _extract_intent(self, q: str) -> str:
        if self._is_top_budget_customer_question(q):
            return "top_budget_customer"
        if self._is_generic_average_question(q):
            return "generic_average"
        if self._is_generic_min_max_question(q):
            return "generic_min_max"
        if any(word in q for word in self.COUNT_WORDS):
            return "count"
        if self._is_generic_top_question(q):
            return "generic_top"
        if self._find_cell_value_matches(q) and self._find_requested_columns(q):
            return "generic_attribute"
        if self._find_customer_name_in_question(q) and self._requested_customer_attribute(q):
            return "customer_attribute"
        if any(word in q for word in self.TOP_WORDS) and self.roles.get("location"):
            return "top_locations"
        if any(word in q for word in self.AVG_WORDS) and self.roles.get("budget"):
            return "average_budget"
        if any(word in q for word in self.MINMAX_WORDS) and self.roles.get("budget"):
            return "budget_min_max"
        if any(word in q for word in self.SUMMARY_WORDS):
            return "summary"
        if any(word in q for word in self.LIST_WORDS):
            return "list"
        return "list"

    def _is_top_budget_customer_question(self, q: str) -> bool:
        budget_col = self.roles.get("budget")
        name_col = self.roles.get("name")
        if not budget_col or not name_col:
            return False

        asks_for_person = any(word in q for word in ("name", "customer", "client", "lead", "who", "which"))
        asks_for_top = any(word in q for word in ("top", "highest", "maximum", "max", "largest", "biggest"))
        asks_for_budget = any(word in q for word in ("budget", "price", "amount", "investment"))
        return asks_for_person and asks_for_top and asks_for_budget

    def _is_generic_average_question(self, q: str) -> bool:
        return any(word in q for word in self.AVG_WORDS) and bool(self._find_requested_columns(q))

    def _is_generic_min_max_question(self, q: str) -> bool:
        return any(word in q for word in self.MINMAX_WORDS) and bool(self._find_requested_columns(q))

    def _is_generic_top_question(self, q: str) -> bool:
        return any(word in q for word in self.TOP_WORDS) and bool(self._find_requested_columns(q))

    def _find_requested_columns(self, q: str) -> list[str]:
        """Return real DataFrame columns explicitly or semantically mentioned."""

        normalized_question = normalize_column_name(q)
        requested: list[str] = []

        for column, normalized_column in self.normalized_columns.items():
            column_phrase = normalized_column.replace("_", " ")
            plural_normalized = f"{normalized_column}s"
            plural_phrase = f"{column_phrase}s"
            if normalized_column and (normalized_column in normalized_question or plural_normalized in normalized_question):
                requested.append(column)
            elif column_phrase and (
                re.search(rf"\b{re.escape(column_phrase)}\b", q)
                or re.search(rf"\b{re.escape(plural_phrase)}\b", q)
            ):
                requested.append(column)

        for role, synonyms in self.ROLE_SYNONYMS.items():
            column = self.roles.get(role)
            if not column:
                continue
            if role == "name":
                explicit_name_words = (
                    "name",
                    "names",
                    "customer name",
                    "customer names",
                    "client name",
                    "client names",
                    "lead name",
                    "lead names",
                    "who",
                )
                if not any(re.search(rf"\b{re.escape(term)}\b", q) for term in explicit_name_words):
                    continue
            if any(
                re.search(rf"\b{re.escape(synonym.replace('_', ' '))}\b", q)
                or re.search(rf"\b{re.escape(synonym.replace('_', ' '))}s\b", q)
                for synonym in synonyms
            ):
                requested.append(column)

        return list(dict.fromkeys(requested))

    def _is_high_intent_summary_question(self, q: str) -> bool:
        has_summary_word = any(word in q for word in self.SUMMARY_WORDS)
        asks_high_intent = "high_intent" in q.replace("-", "_") or "high intent" in q
        return has_summary_word and asks_high_intent

    def _build_high_intent_summary_result(
        self, df: pd.DataFrame, filters: list[str]
    ) -> QueryResult:
        budget_col = self.roles.get("budget")
        location_col = self.roles.get("location")
        property_col = self.roles.get("property_type")

        metrics: dict[str, Any] = {"High-Intent Customers": int(len(df))}
        if budget_col and budget_col in df.columns and df[budget_col].notna().any():
            metrics["Average Budget"] = format_inr(df[budget_col].mean())
        if location_col and location_col in df.columns and df[location_col].notna().any():
            top_location = df[location_col].astype(str).value_counts().idxmax()
            metrics["Top Location"] = top_location
        if property_col and property_col in df.columns and df[property_col].notna().any():
            top_property = df[property_col].astype(str).value_counts().idxmax()
            metrics["Top Property Type"] = top_property

        display_columns = self._default_display_columns()
        answer = f"Summary generated for {len(df)} high-intent customer(s)."
        if df.empty:
            answer = "No high-intent customers were found in the uploaded Excel data."

        return QueryResult(
            answer=answer,
            summary="",
            records=self._prepare_records_for_display(df[display_columns]),
            metrics=metrics,
            intent="summary",
            filters_applied=filters,
        )

    def _find_cell_value_matches(self, q: str) -> list[tuple[str, str]]:
        """Find exact mentions of real cell values across all columns."""

        matches: list[tuple[str, str]] = []
        for column in self.df.columns:
            values = self.df[column].dropna().astype(str).unique()
            for value in values:
                normalized_value = normalize_text(value)
                if not normalized_value or normalized_value in {"nan", "none"}:
                    continue
                if len(normalized_value) < 2:
                    continue
                if re.search(rf"\b{re.escape(normalized_value)}\b", q):
                    matches.append((column, value))

        matches.sort(key=lambda item: len(normalize_text(item[1])), reverse=True)
        deduped: list[tuple[str, str]] = []
        seen_columns: set[str] = set()
        for column, value in matches:
            if column in seen_columns:
                continue
            deduped.append((column, value))
            seen_columns.add(column)
        return deduped

    def _find_customer_name_in_question(self, q: str) -> str | None:
        name_col = self.roles.get("name")
        if not name_col:
            return None

        names = [str(name) for name in self.df[name_col].dropna().unique()]
        names = sorted(names, key=len, reverse=True)
        for name in names:
            normalized_name = normalize_text(name)
            if normalized_name and re.search(rf"\b{re.escape(normalized_name)}\b", q):
                return name
        return None

    def _requested_customer_attribute(self, q: str) -> tuple[str, str] | None:
        attribute_keywords = [
            ("budget", ("budget", "price", "amount", "investment")),
            ("contact", ("contact", "phone", "mobile", "number")),
            ("location", ("location", "area", "locality", "where")),
            ("property_type", ("property type", "property", "configuration", "bhk", "villa")),
            ("possession", ("possession", "handover")),
            ("status", ("status", "call status", "last call")),
        ]
        for role, keywords in attribute_keywords:
            column = self.roles.get(role)
            if column and any(keyword in q for keyword in keywords):
                return role, column
        return None

    def _apply_filters(
        self, df: pd.DataFrame, q: str, filters: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        df = self._apply_generic_value_filters(df, q, filters)
        df = self._apply_name_filter(df, q, filters)
        df = self._apply_budget_filter(df, q, filters)
        df = self._apply_property_filter(df, q, filters)
        df = self._apply_location_filter(df, q, filters)
        df = self._apply_status_filter(df, q, filters)
        return df, filters

    def _apply_generic_value_filters(self, df: pd.DataFrame, q: str, filters: list[str]) -> pd.DataFrame:
        """Filter by exact cell values mentioned in the question, for any column."""

        for column, value in self._find_cell_value_matches(q):
            filter_text = f"{column} = {value}"
            if filter_text in filters:
                continue
            filters.append(filter_text)
            df = df[df[column].astype(str).str.lower() == str(value).lower()]
        return df

    def _apply_name_filter(self, df: pd.DataFrame, q: str, filters: list[str]) -> pd.DataFrame:
        name_col = self.roles.get("name")
        if not name_col:
            return df

        matched_name = self._find_customer_name_in_question(q)
        if not matched_name:
            return df

        filter_text = f"{name_col} = {matched_name}"
        if filter_text not in filters:
            filters.append(filter_text)
        return df[df[name_col].astype(str).str.lower() == matched_name.lower()]

    def _apply_budget_filter(self, df: pd.DataFrame, q: str, filters: list[str]) -> pd.DataFrame:
        budget_col = self.roles.get("budget")
        if not budget_col:
            return df

        between_match = re.search(
            r"between\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)\s+and\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)",
            q,
        )
        if between_match:
            low = parse_money_value(between_match.group(1))
            high = parse_money_value(between_match.group(2))
            if low is not None and high is not None:
                low, high = sorted([low, high])
                filters.append(f"{budget_col} between {format_inr(low)} and {format_inr(high)}")
                return df[df[budget_col].between(low, high, inclusive="both")]

        comparator_patterns = [
            (r"(?:above|over|greater than|more than|higher than)\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)", ">"),
            (r"(?:below|under|less than|lower than)\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)", "<"),
            (r"(?:at least|minimum of)\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)", ">="),
            (r"(?:up to|upto|maximum of)\s+([\d,.]+\s*(?:cr|crore|crores|lakh|lakhs|lac|l|k)?)", "<="),
        ]
        for pattern, operator in comparator_patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            amount = parse_money_value(match.group(1))
            if amount is None:
                continue
            filters.append(f"{budget_col} {operator} {format_inr(amount)}")
            if operator == ">":
                return df[df[budget_col] > amount]
            if operator == "<":
                return df[df[budget_col] < amount]
            if operator == ">=":
                return df[df[budget_col] >= amount]
            if operator == "<=":
                return df[df[budget_col] <= amount]
        return df

    def _apply_property_filter(self, df: pd.DataFrame, q: str, filters: list[str]) -> pd.DataFrame:
        property_col = self.roles.get("property_type")
        if not property_col:
            return df

        values = [str(v) for v in df[property_col].dropna().unique()]
        matched_values = []
        for value in values:
            normalized = normalize_text(value)
            if normalized and normalized in q:
                matched_values.append(value)

        bhk_match = re.search(r"\b([1-9])\s*bhk\b", q)
        if bhk_match:
            bhk = f"{bhk_match.group(1)}BHK"
            matched_values.extend([value for value in values if normalize_text(value) == normalize_text(bhk)])
            if not matched_values:
                filters.append(f"{property_col} contains '{bhk}'")
                return df[df[property_col].astype(str).str.contains(bhk, case=False, na=False)]

        if "villa" in q or "villas" in q:
            matched_values.extend([value for value in values if "villa" in normalize_text(value)])
            if not matched_values:
                filters.append(f"{property_col} contains 'villa'")
                return df[df[property_col].astype(str).str.contains("villa", case=False, na=False)]

        matched_values = sorted(set(matched_values))
        if not matched_values:
            return df

        filters.append(f"{property_col} in {', '.join(matched_values)}")
        mask = df[property_col].astype(str).str.lower().isin([v.lower() for v in matched_values])
        return df[mask]

    def _apply_location_filter(self, df: pd.DataFrame, q: str, filters: list[str]) -> pd.DataFrame:
        location_col = self.roles.get("location")
        if not location_col:
            return df

        values = [str(v) for v in df[location_col].dropna().unique()]
        matched_values = []
        for value in values:
            normalized = normalize_text(value)
            if normalized and re.search(rf"\b{re.escape(normalized)}\b", q):
                matched_values.append(value)

        broad_pune_words = {"pune", "punekar"}
        if any(word in q.split() for word in broad_pune_words) and not matched_values:
            filters.append("Pune interpreted as the uploaded Pune lead dataset, no locality filter applied")
            return df

        city_match = re.search(r"\b(?:from|in|near|at)\b\s+([a-z][a-z\s-]{1,30})", q)
        if city_match and not matched_values:
            requested = city_match.group(1).strip()
            stop_words = {"customers", "customer", "leads", "lead", "with", "budget", "looking", "for"}
            requested = " ".join(word for word in requested.split() if word not in stop_words)
            if requested and requested not in {"pune"} and not self._looks_like_month_phrase(requested):
                filters.append(f"{location_col} contains '{requested}'")
                return df[df[location_col].astype(str).str.contains(requested, case=False, na=False)]

        if not matched_values:
            return df

        filters.append(f"{location_col} in {', '.join(sorted(set(matched_values)))}")
        mask = df[location_col].astype(str).str.lower().isin([v.lower() for v in matched_values])
        return df[mask]

    def _apply_status_filter(self, df: pd.DataFrame, q: str, filters: list[str]) -> pd.DataFrame:
        status_col = self.roles.get("status")
        if not status_col:
            return df

        values = [str(v) for v in df[status_col].dropna().unique()]
        matched_values = [value for value in values if normalize_text(value) in q]
        if not matched_values:
            return df

        filters.append(f"{status_col} in {', '.join(sorted(set(matched_values)))}")
        mask = df[status_col].astype(str).str.lower().isin([v.lower() for v in matched_values])
        return df[mask]

    def _apply_high_intent_filter(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        status_col = self.roles.get("status")
        budget_col = self.roles.get("budget")

        if status_col:
            connected = df[status_col].astype(str).str.contains("connected", case=False, na=False)
            if connected.any():
                return df[connected], f"{status_col} contains Connected"

        if budget_col and df[budget_col].notna().any():
            threshold = df[budget_col].quantile(0.75)
            return df[df[budget_col] >= threshold], f"{budget_col} in top 25% from uploaded data"

        return df, "No explicit intent column found; returned all rows"

    def _filter_columns_from_filters(self, filters: list[str]) -> list[str]:
        columns: list[str] = []
        for filter_text in filters:
            for column in self.df.columns:
                if filter_text.startswith(f"{column} =") and column not in columns:
                    columns.append(column)
        return columns

    def _is_money_column(self, column: str) -> bool:
        normalized = normalize_column_name(column)
        return any(token in normalized for token in ("budget", "price", "amount", "cost", "investment"))

    def _first_numeric_requested_column(self, q: str) -> str | None:
        for column in self._find_requested_columns(q):
            if column not in self.df.columns:
                continue
            numeric_values = coerce_numeric(self.df[column]).dropna()
            if not numeric_values.empty:
                return column
        return None

    def _format_column_value(self, column: str, value: Any) -> str:
        if pd.isna(value):
            return "N/A"
        if self._is_money_column(column):
            return format_inr(value)
        normalized = normalize_column_name(column)
        if any(token in normalized for token in ("contact", "phone", "mobile", "number")):
            return str(value).split(".")[0]
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _prepare_records_for_display(self, df: pd.DataFrame) -> pd.DataFrame:
        display_df = df.copy()
        for column in display_df.columns:
            normalized = normalize_column_name(column)
            if any(token in normalized for token in ("contact", "phone", "mobile", "number")):
                display_df[column] = display_df[column].apply(
                    lambda value: "" if pd.isna(value) else str(value).split(".")[0]
                )
        return display_df

    def _execute(self, intent: str, q: str, df: pd.DataFrame, filters: list[str]) -> QueryResult:
        budget_col = self.roles.get("budget")
        location_col = self.roles.get("location")

        if intent == "generic_attribute":
            return self._execute_generic_attribute(q, df, filters)

        if intent == "generic_average":
            return self._execute_generic_average(q, df, filters)

        if intent == "generic_min_max":
            return self._execute_generic_min_max(q, df, filters)

        if intent == "generic_top":
            return self._execute_generic_top(q, df, filters)

        if intent == "count":
            count = int(len(df))
            return QueryResult(
                answer=f"{count} matching customer(s) found.",
                records=self._prepare_records_for_display(df),
                metrics={"Matching Customers": count},
                intent=intent,
                filters_applied=filters,
                summary=""
            )

        if intent == "average_budget" and budget_col:
            avg = df[budget_col].dropna().mean()
            answer = "No numeric budget values were found for the matching records."
            metrics = {}
            if pd.notna(avg):
                answer = f"The average budget is {format_inr(avg)}."
                metrics = {"Average Budget": format_inr(avg), "Matching Customers": int(len(df))}
            return QueryResult(answer=answer, records=self._prepare_records_for_display(df), metrics=metrics, intent=intent, filters_applied=filters, summary="")

        if intent == "budget_min_max" and budget_col:
            budget_values = df[budget_col].dropna()
            if budget_values.empty:
                answer = "No numeric budget values were found for the matching records."
                metrics = {}
            else:
                min_budget = budget_values.min()
                max_budget = budget_values.max()
                answer = f"The minimum budget is {format_inr(min_budget)} and the maximum budget is {format_inr(max_budget)}."
                metrics = {
                    "Minimum Budget": format_inr(min_budget),
                    "Maximum Budget": format_inr(max_budget),
                    "Matching Customers": int(len(df)),
                }
            return QueryResult(answer=answer, records=self._prepare_records_for_display(df), metrics=metrics, intent=intent, filters_applied=filters, summary="")

        if intent == "customer_attribute":
            return self._execute_customer_attribute(q, df, filters)

        if intent == "top_budget_customer" and budget_col:
            name_col = self.roles.get("name")
            if df.empty or df[budget_col].dropna().empty:
                return QueryResult(
                    answer="No customer with a numeric budget was found in the matching records.",
                    records=self._prepare_records_for_display(df),
                    metrics={"Matching Customers": int(len(df))},
                    intent=intent,
                    filters_applied=filters,
                    summary=""
                )

            top_row = df.sort_values(by=budget_col, ascending=False, na_position="last").head(1)
            top_budget = top_row.iloc[0][budget_col]
            if name_col and name_col in top_row.columns:
                top_name = top_row.iloc[0][name_col]
                answer = f"{top_name} is the customer with the highest budget of {format_inr(top_budget)}."
                metrics = {"Top Customer": top_name, "Highest Budget": format_inr(top_budget)}
            else:
                answer = f"The highest customer budget is {format_inr(top_budget)}."
                metrics = {"Highest Budget": format_inr(top_budget)}

            return QueryResult(
                answer=answer,
                records=top_row,
                metrics=metrics,
                intent=intent,
                filters_applied=filters,
                summary=""
            )

        if intent == "top_locations" and location_col:
            top_n = self._extract_top_n(q)
            grouped = (
                df[location_col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(top_n)
                .rename_axis(location_col)
                .reset_index(name="Customer Interest")
            )
            answer = f"Top {len(grouped)} location(s) by customer interest are shown below."
            if not grouped.empty:
                first = grouped.iloc[0]
                answer = f"{first[location_col]} has the highest interest with {int(first['Customer Interest'])} customer(s)."
            return QueryResult(
                answer=answer,
                records=grouped,
                chart_data=grouped.set_index(location_col) if not grouped.empty else grouped,
                metrics={"Locations": int(len(grouped)), "Matching Customers": int(len(df))},
                intent=intent,
                filters_applied=filters,
                summary=""
            )

        if intent == "summary":
            records = self._summary_records(df)
            answer = f"Summary generated from {len(df)} matching customer record(s)."
            return QueryResult(
                answer=answer,
                records=records,
                metrics={"Matching Customers": int(len(df))},
                intent=intent,
                filters_applied=filters,
                summary=""
            )

        sorted_df = self._sort_for_display(df, q)
        return QueryResult(
            answer=f"{len(sorted_df)} matching customer record(s) found.",
            records=self._prepare_records_for_display(sorted_df),
            metrics={"Matching Customers": int(len(sorted_df))},
            intent="list",
            filters_applied=filters,
            summary=""
        )

    def _execute_generic_attribute(
        self, q: str, df: pd.DataFrame, filters: list[str]
    ) -> QueryResult:
        requested_columns = self._find_requested_columns(q)
        filter_columns = self._filter_columns_from_filters(filters)

        display_columns = filter_columns + [col for col in requested_columns if col not in filter_columns]
        if not display_columns:
            display_columns = list(df.columns)
        display_columns = [column for column in display_columns if column in df.columns]

        if df.empty:
            return QueryResult(
                answer="No matching records were found in the uploaded Excel data.",
                records=self._prepare_records_for_display(df[display_columns] if display_columns else df),
                metrics={"Matching Records": 0},
                intent="generic_attribute",
                filters_applied=filters,
                summary=""
            )

        answer_parts = []
        for column in requested_columns:
            if column not in df.columns:
                continue
            values = df[column].dropna()
            if values.empty:
                answer_parts.append(f"{column}: no value available")
                continue

            unique_values = [self._format_column_value(column, value) for value in values.drop_duplicates()]

            if len(unique_values) <= 5:
                answer_parts.append(f"{column}: {', '.join(unique_values)}")
            else:
                answer_parts.append(f"{column}: {len(unique_values)} unique values")

        answer = f"Found {len(df)} matching record(s)."
        if answer_parts:
            answer = f"Found {len(df)} matching record(s). " + " | ".join(answer_parts)

        return QueryResult(
            answer=answer,
            records=self._prepare_records_for_display(df[display_columns]),
            metrics={"Matching Records": int(len(df))},
            intent="generic_attribute",
            filters_applied=filters,
            summary=""
        )

    def _execute_generic_average(
        self, q: str, df: pd.DataFrame, filters: list[str]
    ) -> QueryResult:
        column = self._first_numeric_requested_column(q)
        if not column:
            return QueryResult(
                answer="No numeric column was detected for the average calculation.",
                records=self._prepare_records_for_display(df),
                metrics={"Matching Records": int(len(df))},
                intent="generic_average",
                filters_applied=filters,
                summary=""
            )

        values = coerce_numeric(df[column]).dropna()
        if values.empty:
            answer = f"No numeric values were found in {column} for the matching records."
            metrics = {"Matching Records": int(len(df))}
        else:
            avg = values.mean()
            formatted = format_inr(avg) if self._is_money_column(column) else f"{avg:,.2f}"
            answer = f"The average {column} is {formatted}."
            metrics = {f"Average {column}": formatted, "Matching Records": int(len(df))}

        return QueryResult(
            answer=answer,
            records=self._prepare_records_for_display(df),
            metrics=metrics,
            intent="generic_average",
            filters_applied=filters,
            summary=""
        )

    def _execute_generic_min_max(
        self, q: str, df: pd.DataFrame, filters: list[str]
    ) -> QueryResult:
        column = self._first_numeric_requested_column(q)
        if not column:
            return QueryResult(
                answer="No numeric column was detected for the min/max calculation.",
                records=self._prepare_records_for_display(df),
                metrics={"Matching Records": int(len(df))},
                intent="generic_min_max",
                filters_applied=filters,
                summary=""
            )

        values = coerce_numeric(df[column]).dropna()
        if values.empty:
            answer = f"No numeric values were found in {column} for the matching records."
            metrics = {"Matching Records": int(len(df))}
        else:
            min_value = values.min()
            max_value = values.max()
            min_text = format_inr(min_value) if self._is_money_column(column) else f"{min_value:g}"
            max_text = format_inr(max_value) if self._is_money_column(column) else f"{max_value:g}"
            answer = f"The minimum {column} is {min_text} and the maximum {column} is {max_text}."
            metrics = {f"Minimum {column}": min_text, f"Maximum {column}": max_text}

        return QueryResult(
            answer=answer,
            records=self._prepare_records_for_display(df),
            metrics=metrics,
            intent="generic_min_max",
            filters_applied=filters,
            summary=""
        )

    def _execute_generic_top(
        self, q: str, df: pd.DataFrame, filters: list[str]
    ) -> QueryResult:
        requested_columns = self._find_requested_columns(q)
        top_n = self._extract_top_n(q)
        numeric_column = self._first_numeric_requested_column(q)

        if numeric_column:
            sorted_df = df.sort_values(by=numeric_column, ascending=False, na_position="last").head(top_n)
            value = sorted_df.iloc[0][numeric_column] if not sorted_df.empty else None
            formatted = format_inr(value) if self._is_money_column(numeric_column) else value
            answer = f"Top {len(sorted_df)} record(s) by {numeric_column} are shown below."
            if len(sorted_df) == 1 and value is not None:
                answer = f"The top {numeric_column} value is {formatted}."
            return QueryResult(
                answer=answer,
                records=self._prepare_records_for_display(sorted_df),
                metrics={f"Top {numeric_column}": formatted, "Matching Records": int(len(df))},
                intent="generic_top",
                filters_applied=filters,
                summary=""
            )

        group_column = requested_columns[0] if requested_columns else None
        if group_column and group_column in df.columns:
            grouped = (
                df[group_column]
                .dropna()
                .astype(str)
                .value_counts()
                .head(top_n)
                .rename_axis(group_column)
                .reset_index(name="Count")
            )
            answer = f"Top {len(grouped)} {group_column} value(s) by count are shown below."
            return QueryResult(
                answer=answer,
                records=grouped,
                chart_data=grouped.set_index(group_column) if not grouped.empty else grouped,
                metrics={"Matching Records": int(len(df))},
                intent="generic_top",
                filters_applied=filters,
                summary=""
            )

        return QueryResult(
            answer=f"{len(df)} matching customer record(s) found.",
            records=self._prepare_records_for_display(df),
            metrics={"Matching Records": int(len(df))},
            intent="generic_top",
            filters_applied=filters,
            summary=""
        )

    def _execute_customer_attribute(
        self, q: str, df: pd.DataFrame, filters: list[str]
    ) -> QueryResult:
        requested = self._requested_customer_attribute(q)
        name_col = self.roles.get("name")
        if not requested or not name_col:
            return QueryResult(
                answer=f"{len(df)} matching customer record(s) found.",
                records=self._prepare_records_for_display(df),
                metrics={"Matching Customers": int(len(df))},
                intent="customer_attribute",
                filters_applied=filters,
                summary=""
            )

        role, attribute_col = requested
        if df.empty:
            return QueryResult(
                answer="No matching customer records were found in the uploaded Excel data.",
                records=self._prepare_records_for_display(df),
                metrics={"Matching Customers": 0},
                intent="customer_attribute",
                filters_applied=filters,
                summary=""
            )

        customer_names = [str(name) for name in df[name_col].dropna().unique()]
        customer_label = customer_names[0] if len(customer_names) == 1 else "The matching customers"
        values = df[attribute_col].dropna()

        if values.empty:
            answer = f"No {attribute_col} value is available for {customer_label}."
            metrics = {"Matching Records": int(len(df))}
        elif role == "budget":
            formatted_values = [format_inr(value) for value in values]
            unique_values = list(dict.fromkeys(formatted_values))
            if len(unique_values) == 1:
                answer = f"{customer_label}'s budget is {unique_values[0]}."
            else:
                answer = (
                    f"{customer_label} has {len(df)} matching records with budgets: "
                    f"{', '.join(unique_values)}."
                )
            metrics = {
                "Matching Records": int(len(df)),
                "Minimum Budget": format_inr(values.min()),
                "Maximum Budget": format_inr(values.max()),
            }
        else:
            unique_values = [str(value) for value in values.astype(str).drop_duplicates()]
            if len(unique_values) == 1:
                answer = f"{customer_label}'s {attribute_col} is {unique_values[0]}."
            else:
                answer = (
                    f"{customer_label} has {len(df)} matching records with {attribute_col} values: "
                    f"{', '.join(unique_values)}."
                )
            metrics = {"Matching Records": int(len(df)), "Unique Values": int(len(unique_values))}

        display_columns = [name_col, attribute_col]
        for role_name in ("budget", "property_type", "location", "contact", "possession", "status"):
            column = self.roles.get(role_name)
            if column and column not in display_columns:
                display_columns.append(column)
        display_columns = [column for column in display_columns if column in df.columns]

        return QueryResult(
            answer=answer,
            records=self._prepare_records_for_display(df[display_columns]),
            metrics=metrics,
            intent="customer_attribute",
            filters_applied=filters,
            summary=""
        )

    def _extract_top_n(self, q: str) -> int:
        match = re.search(r"top\s+(\d+)", q)
        if match:
            return max(1, min(int(match.group(1)), 25))
        return 5

    def _sort_for_display(self, df: pd.DataFrame, q: str) -> pd.DataFrame:
        budget_col = self.roles.get("budget")
        if not budget_col or budget_col not in df.columns:
            return df
        if "lowest" in q or "minimum" in q or "cheapest" in q:
            return df.sort_values(by=budget_col, ascending=True, na_position="last")
        return df.sort_values(by=budget_col, ascending=False, na_position="last")

    def _summary_records(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        rows.append({"Metric": "Matching Customers", "Value": len(df)})

        budget_col = self.roles.get("budget")
        if budget_col and df[budget_col].notna().any():
            rows.extend(
                [
                    {"Metric": "Average Budget", "Value": format_inr(df[budget_col].mean())},
                    {"Metric": "Minimum Budget", "Value": format_inr(df[budget_col].min())},
                    {"Metric": "Maximum Budget", "Value": format_inr(df[budget_col].max())},
                ]
            )

        for role in ("property_type", "location", "status"):
            column = self.roles.get(role)
            if column and column in df.columns and df[column].notna().any():
                top_value = df[column].astype(str).value_counts().idxmax()
                top_count = int(df[column].astype(str).value_counts().max())
                rows.append({"Metric": f"Top {column}", "Value": f"{top_value} ({top_count})"})

        return pd.DataFrame(rows)

    def _generate_final_answer(self, question: str, result: QueryResult) -> str:
        """Use Gemini to write the final answer from already-computed facts.

        This does not change query intent, filters, records, metrics, charts, or
        any Pandas execution logic. It only rewrites the final answer text.
        """

        if not self.gemini.available:
            return result.answer

        filters_text = "; ".join(result.filters_applied) if result.filters_applied else "No filters"
        records = dataframe_to_records(result.records, max_rows=10)
        prompt = (
            "You are the final response writer for a customer-data chatbot.\n"
            "The data query has already been executed using Pandas. "
            "Write one concise, natural, human-like answer using only the computed facts below.\n\n"
            "Rules:\n"
            "- Do not run a new analysis.\n"
            "- Do not change the intent.\n"
            "- Do not invent names, locations, budgets, counts, or recommendations.\n"
            "- If there are zero matching records, clearly say no records matched those filters.\n"
            "- If the computed result does not contain the fact the user asked for, say what the computed result shows.\n"
            "- Keep the answer to 1-2 sentences.\n\n"
            f"User question: {question}\n"
            f"Query intent: {result.intent}\n"
            f"Applied filters: {filters_text}\n"
            f"Fallback factual answer: {result.answer}\n"
            f"Computed metrics: {result.metrics}\n"
            f"Computed result records: {records}\n"
        )
        generated_answer = self.gemini.generate(prompt)
        return generated_answer or result.answer

    def _summarize(self, question: str, result: QueryResult) -> str:
        filters_text = "; ".join(result.filters_applied) if result.filters_applied else "No filters"
        displayed_rows = len(result.records)
        if displayed_rows == 0:
            base_summary = (
                f"No rows in the uploaded Excel file matched the requested conditions: {filters_text}. "
                "The result is based on dataframe filtering, so no matching table rows can be shown."
            )
        elif filters_text == "No filters":
            base_summary = (
                f"The result was computed from the uploaded Excel data and {displayed_rows} row(s) are displayed. "
                "The answer uses the matching dataframe records only."
            )
        else:
            base_summary = (
                f"The result was computed after applying: {filters_text}. "
                f"{displayed_rows} matching row(s) are displayed from the uploaded Excel data."
            )

        prompt = (
            "Write a concise 2-4 line business summary for the user. "
            "Use only the computed facts below. Do not mention internal query intent, implementation details, "
            "Pandas, prompts, or dataframe mechanics. Do not add assumptions or recommendations.\n\n"
            f"Question: {question}\n"
            f"Answer: {result.answer}\n"
            f"Filters: {filters_text}\n"
            f"Metrics: {result.metrics}\n"
            f"Displayed records sample: {dataframe_to_records(result.records, max_rows=5)}"
        )
        llm_summary = self.gemini.generate(prompt)
        return llm_summary or base_summary

    def schema_report(self) -> pd.DataFrame:
        """Return detected schema roles for display/debugging."""

        rows = []
        for role, column in self.roles.items():
            rows.append({"Detected Role": role, "Column": column or "Not detected"})
        return pd.DataFrame(rows)
