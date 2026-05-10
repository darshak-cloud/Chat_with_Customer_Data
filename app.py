"""Streamlit frontend for Chat with Customer Data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from query_engine import CustomerDataQueryEngine, QueryResult
from utils import GeminiHelper, load_excel_file, safe_preview


st.set_page_config(
    page_title="Chat with Customer Data",
    layout="wide",
    initial_sidebar_state="expanded",
)


SAMPLE_QUESTIONS = [
    "How many customers have budget above 90 lakhs?",
    "List customers looking for 2BHK in Pune",
    "What is the average budget?",
    "Give a summary of all high-intent customers",
    "Show top 5 locations by customer interest",
    "Which customers are interested in villas?",
    "How many leads came from Mumbai?",
    "What is the minimum and maximum budget?",
    "Show all customers with budget between 70L and 1Cr",
]


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("engine", None)


def build_engine(df: pd.DataFrame) -> CustomerDataQueryEngine:
    return CustomerDataQueryEngine(df=df, gemini=GeminiHelper())


def render_metrics(result: QueryResult) -> None:
    if not result.metrics:
        return
    cols = st.columns(min(4, len(result.metrics)))
    for col, (label, value) in zip(cols, result.metrics.items()):
        col.metric(label, value)


def render_result(result: QueryResult) -> None:
    st.markdown("**Answer**")
    st.write(result.answer)

    st.markdown("**Summary**")
    st.info(result.summary)

    render_metrics(result)

    if result.chart_data is not None and not result.chart_data.empty:
        st.markdown("**Chart**")
        st.bar_chart(result.chart_data)

    if result.records is not None and not result.records.empty:
        st.markdown("**Matching Records**")
        st.dataframe(result.records, use_container_width=True, hide_index=True)
    else:
        st.warning("No matching records were found in the uploaded Excel data.")


def run_query(question: str) -> None:
    engine = st.session_state.get("engine")
    if engine is None:
        st.warning("Please upload an Excel file first.")
        return

    result = engine.answer_question(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.session_state.history.append({"question": question, "answer": result.answer})


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Chat with Customer Data")
        st.caption("Upload an Excel file and ask factual questions powered by Pandas and Gemini.")

        uploaded_file = st.file_uploader("Upload customer Excel file", type=["xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df = load_excel_file(uploaded_file)
                st.session_state.df = df
                st.session_state.engine = build_engine(df)
                st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
            except Exception as exc:
                st.error(f"Could not read Excel file: {exc}")

        if st.session_state.df is not None:
            df = st.session_state.df
            st.divider()
            st.subheader("Dataset Metrics")
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))

            gemini = st.session_state.engine.gemini
            if gemini.available:
                st.success("Gemini connected for query planning, final answers, and summaries.")
            else:
                st.warning(f"Gemini not connected: {gemini.last_error or 'unknown error'}")

            with st.expander("Detected Schema", expanded=False):
                st.dataframe(st.session_state.engine.schema_report(), hide_index=True, use_container_width=True)

            with st.expander("Query History", expanded=True):
                if st.session_state.history:
                    for item in reversed(st.session_state.history[-8:]):
                        st.markdown(f"**Q:** {item['question']}")
                        st.caption(item["answer"])
                else:
                    st.caption("No questions asked yet.")

        st.divider()
        st.subheader("Suggested Questions")
        for question in SAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                run_query(question)
                st.rerun()


def render_main() -> None:
    st.title("Chat with Customer Data")
    st.write("Ask natural-language questions. Gemini plans the query and writes the response, while Pandas executes the validated data operations.")

    if st.session_state.df is None:
        st.info("Upload an Excel file from the sidebar to begin.")
        return

    with st.expander("Data Preview", expanded=True):
        st.dataframe(safe_preview(st.session_state.df), use_container_width=True, hide_index=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and isinstance(message["content"], QueryResult):
                render_result(message["content"])
            else:
                st.write(message["content"])

    prompt = st.chat_input("Ask a question about the customer data")
    if prompt:
        run_query(prompt)
        st.rerun()


def main() -> None:
    init_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
