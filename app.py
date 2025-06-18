import streamlit as st
import pandas as pd
import duckdb
import io
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import plotly.graph_objects as go
from datetime import datetime
import re

# Load OpenAI API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Smart Adhoc Agent", layout="wide")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Smart Adhoc Agent")

# File input
with st.sidebar:
    st.header("📁 Upload Data")
    file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])
    sheet_url = st.text_input("Paste public Google Sheets URL")

# Load and register data
tables = {}
df_preview = None
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        tables["data"] = df
    elif file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            sheet_df = xls.parse(sheet)
            table_name = sheet.lower().replace(" ", "_").replace("-", "_")
            tables[table_name] = sheet_df
    st.success(f"✅ Loaded {len(tables)} table(s): {', '.join(tables.keys())}")
elif sheet_url and "docs.google.com" in sheet_url:
    try:
        csv_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
        data = requests.get(csv_url).content
        df = pd.read_csv(io.StringIO(data.decode("utf-8")))
        tables["data"] = df
        st.success("✅ Google Sheet loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read Google Sheet: {e}")

# Convert possible date-like string columns to datetime for DuckDB compatibility
def is_date_column(series):
    # Heuristic: check if >50% values parse as datetime
    parsed = pd.to_datetime(series, errors='coerce')
    non_null_ratio = parsed.notnull().sum() / len(parsed)
    return non_null_ratio > 0.5

for name, df in tables.items():
    for col in df.columns:
        if df[col].dtype == object:
            if is_date_column(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass

# Register with DuckDB and show preview
if tables:
    con = duckdb.connect()
    for name, df in tables.items():
        try:
            con.register(name, df)
        except Exception as e:
            st.warning(f"❌ Failed to register {name}: {e}")

    # Attempt automatic merge on common columns
    common_cols = set.intersection(*(set(df.columns) for df in tables.values())) if len(tables) > 1 else set()
    if common_cols:
        try:
            table_list = list(tables.keys())
            join_expr = f" USING ({', '.join(common_cols)}) "
            merged_sql = f"SELECT * FROM {table_list[0]} " + " ".join([f"JOIN {tbl} {join_expr}" for tbl in table_list[1:]])
            merged_df = con.sql(merged_sql).df()
            con.register("merged_data", merged_df)
            tables["merged_data"] = merged_df
            st.success(f"🧬 Auto-merged table 'merged_data' created on common columns: {', '.join(common_cols)}")
        except Exception as e:
            st.warning(f"⚠️ Failed to auto-merge tables: {e}")

    # Show preview and profiling for first table
    first_key = next(iter(tables))
    df_preview = tables[first_key]
    st.subheader("📊 Preview of first table")
    st.dataframe(df_preview.head())

    for name, df in tables.items():
        st.subheader(f"📊 Profiling: {name}")
        profile = df.describe(include='all').transpose()
        st.dataframe(profile)

# User chat input and response handling
if tables:
    user_input = st.chat_input("Ask a question about your data")
    if user_input:
        # Log user query with timestamp
        with open("user_logs.csv", "a") as f:
            f.write(f"{user_input},{datetime.now()}\n")

        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        try:
            schema_description = "\n".join([
                f"{name} → {', '.join(df.columns)}" for name, df in tables.items()
            ])

            prompt = f"""
You are an expert SQL assistant using DuckDB.
Only use the following tables and columns exactly as listed:
{schema_description}

Important:
- Date and datetime columns are properly typed as DATE or TIMESTAMP.
- Use DuckDB-compatible functions for dates, such as EXTRACT(YEAR FROM date_column) or YEAR(date_column).
- When filtering or grouping by year or month, use appropriate date functions.
- Do NOT guess column names.
- Output ONLY the valid SQL query with no explanations or extra text.

User Question:
{user_input}
"""

            history = [
                {"role": r, "content": m if isinstance(m, str) else "<table result>"}
                for r, m in st.session_state.chat_history[-4:]
            ]

            messages = [
                {"role": "system", "content": "You are a SQL expert helping users query data using DuckDB. Only use valid table/column names provided."}
            ] + history + [
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )

            content = response.choices[0].message.content
            match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
            sql_code = match.group(1).strip() if match else content.strip()

            st.chat_message("assistant").markdown(f"💡 SQL Query:\n```sql\n{sql_code}\n```")

            result_df = con.sql(sql_code).df()

            clean_df = result_df.copy()
            if clean_df.index.name or clean_df.index.to_list() == list(range(len(clean_df))):
                clean_df.reset_index(drop=True, inplace=True)

            for col in clean_df.select_dtypes(include=["datetime64[ns]"]).columns:
                clean_df[col] = clean_df[col].dt.strftime('%Y-%m')

            table = go.Figure(data=[go.Table(
                header=dict(values=list(clean_df.columns), fill_color='lightgray', align='left'),
                cells=dict(values=[clean_df[col] for col in clean_df.columns], align='left')
            )])
            st.chat_message("assistant").plotly_chart(table, use_container_width=True, key=f"plot_{len(st.session_state.chat_history)}")

            st.session_state.chat_history.append(("assistant", clean_df))

        except Exception as e:
            st.chat_message("assistant").write(f"⚠️ GPT or SQL Error: {e}")

    # Show chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            if isinstance(msg, pd.DataFrame):
                table = go.Figure(data=[go.Table(
                    header=dict(values=list(msg.columns), fill_color='lightgray', align='left'),
                    cells=dict(values=[msg[col] for msg in msg.columns], align='left')
                )])
                st.chat_message("assistant").plotly_chart(table, use_container_width=True, key=f"history_plot_{hash(str(msg))}")
            else:
                st.chat_message("assistant").write(str(msg))
else:
    st.chat_input("Please upload a file to begin.")
