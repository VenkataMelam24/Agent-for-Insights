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
def load_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=load_api_key())

st.set_page_config(page_title="Smart Adhoc Agent", layout="wide")

# Constants\NMAX_HISTORY = 10  # Max chat history to keep

# Cache duckdb connection\N@st.cache_resource
def get_connection():
    return duckdb.connect()

# Cache data load and parse\N@st.cache_data(show_spinner=False)
def load_and_prepare(file, sheet_url):
    tables = {}
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
    elif sheet_url and "docs.google.com" in sheet_url:
        try:
            csv_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
            data = requests.get(csv_url).content
            df = pd.read_csv(io.StringIO(data.decode("utf-8")))
            tables["data"] = df
        except:
            pass

    # Convert date-like columns to datetime
def is_date_column(series):
    parsed = pd.to_datetime(series, errors='coerce')
    return parsed.notnull().sum() / len(parsed) > 0.5

    for name, df in tables.items():
        for col in df.columns:
            if df[col].dtype == object and is_date_column(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
    return tables

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Smart Adhoc Agent")

# File input
with st.sidebar:
    st.header("📁 Upload Data")
    file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])
    sheet_url = st.text_input("Paste public Google Sheets URL")
    # On-demand profiling
    profile_btn = st.checkbox("Show profiling and merge options")

# Load data once
tables = load_and_prepare(file, sheet_url)

# Connection
con = get_connection()

# Lazy merge
merged_done = False
if profile_btn and tables:
    if st.button("Auto-merge tables on common columns"):
        common_cols = set.intersection(*(set(df.columns) for df in tables.values())) if len(tables)>1 else set()
        if common_cols:
            table_list = list(tables.keys())
            join_expr = f" USING ({', '.join(common_cols)})"
            merged_sql = f"SELECT * FROM {table_list[0]} " + " ".join([f"JOIN {tbl} {join_expr}" for tbl in table_list[1:]])
            merged_df = con.sql(merged_sql).df()
            con.register("merged_data", merged_df)
            tables["merged_data"] = merged_df
            st.success(f"Merged on {', '.join(common_cols)}")
            merged_done = True
    if st.button("Show profiling for all tables"):
        for name, df in tables.items():
            st.subheader(f"Profiling: {name}")
            st.dataframe(df.describe(include='all').transpose())

# Preview first table
if tables:
    first_key = next(iter(tables))
    st.subheader("Preview of first table")
    st.dataframe(tables[first_key].head())

# Chat input & handling
if tables:
    user_input = st.chat_input("Ask a question about your data")
    if user_input:
        # Log user query
        with open("user_logs.csv", "a") as f:
            f.write(f"{user_input},{datetime.now()}\n")

        # Trim history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]

        # Construct prompt
        schema_description = "\n".join(f"{n} -> {', '.join(df.columns)}" for n, df in tables.items())
        prompt = f"""
You are an expert SQL assistant using DuckDB.
Only use the following tables and columns:
{schema_description}
Important:
- Date columns are DATE/TIMESTAMP. Use EXTRACT or YEAR() as needed.
- Output ONLY the SQL query, no explanation.
User Question:
{user_input}
"""
        messages = [{"role":"system","content":"You are..."}] + [
            {"role":r, "content": m if isinstance(m, str) else "<table>"} for r, m in st.session_state.chat_history
        ] + [{"role":"user","content":prompt}]
        response = client.chat.completions.create(model="gpt-4", messages=messages)
        content = response.choices[0].message.content
        match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL|re.IGNORECASE)
        sql_code = match.group(1).strip() if match else content.strip()

        st.chat_message("assistant").markdown(f"💡 SQL Query:\n```sql\n{sql_code}\n```")
        result_df = con.sql(sql_code).df()
        if result_df.empty:
            st.warning("No results found.")
        else:
            clean_df = result_df.reset_index(drop=True)
            for col in clean_df.select_dtypes(include=["datetime64[ns]"]):
                clean_df[col] = clean_df[col].dt.strftime('%Y-%m')
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(clean_df.columns)),
                cells=dict(values=[clean_df[c] for c in clean_df.columns])
            )])
            st.chat_message("assistant").plotly_chart(fig, use_container_width=True)
            st.session_state.chat_history.append(("assistant", clean_df))

    # Render history
    for role, msg in st.session_state.chat_history:
        if role=="assistant" and isinstance(msg, pd.DataFrame):
            st.write(msg)
        elif role=="assistant":
            st.write(msg)
        else:
            st.write(msg)
else:
    st.info("Upload a file to begin.")
