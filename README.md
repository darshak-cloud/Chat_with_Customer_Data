# Chat with Customer Data

A production-ready internship assignment project that lets users upload Excel customer/property lead data and ask natural-language questions. The app answers with factual Pandas-backed results, short summaries, optional charts, and matching records.

## Project Overview

This project follows a Gemini-assisted, data-first architecture:

```text
User Question -> Gemini Query Plan + Schema Planner -> Validated DataFrame Filtering -> Gemini Final Answer -> Gemini Summary
```

Gemini proposes the query plan and writes the final answer/summary. Every selected column and filter is validated against the real DataFrame schema and values before execution. Python and Pandas perform all filtering, aggregation, grouping, sorting, and record retrieval so the answer remains data-backed.

## Features

- Streamlit file upload for `.xlsx` and `.xls` files
- Automatic schema detection for customer lead columns
- Natural-language query handling with a dynamic schema-aware planner, Gemini-assisted planning, fuzzy matching, and rule-based fallbacks
- Pandas execution layer for factual answers
- Budget parsing for values such as `90 lakhs`, `70L`, and `1Cr`
- Count, average, min/max, grouping, top-N, text search, and list queries
- Generic lookup across all uploaded Excel columns and exact cell values
- Automatic date-like column detection for questions such as possession month/year
- Fuzzy matching for close column/value wording
- Query history and chat memory
- Data preview and detected schema view
- Tables, and bar charts
- Graceful fallback messages for invalid or unsupported questions
- Gemini integration using `GEMINI_API_KEY`

## Tech Stack

- Python
- Pandas
- Streamlit
- OpenPyXL
- python-dotenv
- Google Gen AI SDK for Gemini-compatible LLM calls

## Folder Structure

```text
chat-with-customer-data/
|-- app.py
|-- query_engine.py
|-- utils.py
|-- sample_data/
|   |-- README.md
|   `-- pune_real_estate_dummy.xlsx
|-- requirements.txt
|-- sample.env
|-- .gitignore
`-- README.md
```

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your environment file:

```bash
copy sample.env .env
```

4. Add your Gemini API key in `.env`:

```text
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```


## Run Locally

From the project folder:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Upload the Excel file from the sidebar.
2. Review the data preview and detected schema.
3. Ask a question in the chat box or click a suggested question.
4. Review the answer, summary, metrics, charts, and matching records.

For quick testing, use the dummy workbook at:

```text
sample_data/pune_real_estate_dummy.xlsx
```

## Example Queries

- How many customers have budget above 90 lakhs?
- List customers looking for 2BHK in Pune.
- What is the average budget?
- Give a summary of all high-intent customers.
- Show top 5 locations by customer interest.
- Which customers are interested in villas?
- How many leads came from Mumbai?
- What is the minimum and maximum budget?
- Show all customers with budget between 70L and 1Cr.

## Query Engine Design

The engine is implemented in `query_engine.py` and uses these layers:

- Intent extraction: asks Gemini for structured JSON hints, then validates them and falls back to rule-based parsing when needed.
- Column mapping: detects roles such as name, budget, property type, location, contact, possession, and status.
- Query translation assistance: Gemini suggests property, location, status, budget, and top-N filters, but only validated suggestions are applied.
- Generic schema-aware matching: detects real column names and exact cell values from any uploaded Excel column, then uses them as filters or requested output fields.
- Dynamic planner: creates a validated query plan with selected columns, filters, aggregation intent, date/month filters, and fuzzy value matches before falling back to older specialized handlers.
- Pandas execution: applies filters and aggregations using DataFrame operations.
- Result validation: returns empty-state messages when no matching rows exist.
- Final answer generation: Gemini rewrites the computed result into a natural 1-2 sentence answer.
- Summary generation: Gemini writes a concise human-readable summary. Local fallback text is used only if Gemini is not configured or the API call fails.


## Notes on Hallucination Prevention

- Raw Excel data is never passed to Gemini for direct answering.
- All numeric answers are calculated with Pandas.
- Matching records are selected from the uploaded DataFrame.
- Gemini intent hints are treated as suggestions and are validated against actual columns, values, and parseable numeric amounts.
- Gemini final answers are generated only from computed metrics and result rows.
- Gemini summaries receive only already-computed factual results and are instructed not to add new facts.

## Future Improvements

- Add more domain-specific synonyms for lead source, project name, and sales owner columns.
- Add multi-sheet Excel selection.
- Add downloadable query results.
- Add advanced date filters for possession and call time.
- Add authentication for deployed usage.
