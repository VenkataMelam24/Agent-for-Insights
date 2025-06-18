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

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Smart Adhoc Agent", layout="wide")

# Constants
MAX_HISTORY = 10

# Cache DuckDB connection
@st.cache_resource
def get_connection():
    return duckdb.connect()

# Cache data loading and date parsing
@st.cache_data(show_spinner=False)
def load_and_prepare(file, sheet_url):
    tables = {}
    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            tables["data"] = df
        else:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                name = sheet.lower().replace(" ", "_").replace("-", "_")
                tables[name] = df
    elif sheet_url and "docs.google.com" in sheet_url:
        try:
            csv_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
            content = requests.get(csv_url).content
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            tables["data"] = df
        except Exception:
            pass
    # Convert date-like strings to datetime
    def is_date(series):
        parsed = pd.to_datetime(series, errors='coerce')
        return parsed.notnull().sum() / len(parsed) > 0.5
    for name, df in tables.items():
        for col in df.columns:
            if df[col].dtype == object and is_date(df[col]):
                tables[name][col] = pd.to_datetime(df[col], errors='coerce')
    return tables

# Initialize session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Smart Adhoc Agent")

# Sidebar: upload
with st.sidebar:
    st.header("📁 Upload Data")
    file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])
    sheet_url = st.text_input("Paste public Google Sheets URL")
    profiling = st.checkbox("Enable profiling & merge options")

# Load data & get connection
tables = load_and_prepare(file, sheet_url)
con = get_connection()

# Profiling & merge UI
if profiling and tables:
    if st.button("Show Table Profiling"):
        for name, df in tables.items():
            st.subheader(f"Profiling: {name}")
            st.dataframe(df.describe(include='all').transpose())
    if len(tables) > 1 and st.button("Auto-merge Tables"):
        common = set.intersection(*(set(df.columns) for df in tables.values()))
        if common:
            names = list(tables.keys())
            expr = f"USING ({', '.join(common)})"
            sql = f"SELECT * FROM {names[0]} " + " ".join([f"JOIN {t} {expr}" for t in names[1:]])
            merged = con.execute(sql).df()
            tables['merged_data'] = merged
            con.register('merged_data', merged)
            st.success(f"Merged on: {', '.join(common)}")

# Preview
if tables:
    first = next(iter(tables.keys()))
    st.subheader(f"Preview: {first}")
    st.dataframe(tables[first].head())
else:
    st.info("Upload a file or enter a Google Sheet URL to start.")

# Chat interface
if tables:
    user_input = st.chat_input("Ask a question about your data")
    if user_input:
        # Log and trim history
        with open('user_logs.csv', 'a') as log:
            log.write(f"{datetime.now()}\t{user_input}\n")
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
        st.chat_message("user").write(user_input)

        # Prepare prompt
        schema = "\n".join(f"{n}: {', '.join(df.columns)}" for n, df in tables.items())
        prompt = (
            "You are an expert SQL assistant using DuckDB.\n"
            f"Tables and columns:\n{schema}\n"
            "Important:\n"
            "- Date columns are DATE/TIMESTAMP; use EXTRACT(YEAR FROM ...) or YEAR().\n"
            "- Output ONLY the SQL query; no extra text.\n"
            f"Question:\n{user_input}\n"
        )

        # Build messages\        messages = [
            {"role": "system", "content": "You are a SQL expert for DuckDB."}
        ] + [
            {"role": role, "content": msg if isinstance(msg, str) else '<table>'}
            for role, msg in st.session_state.chat_history
        ] + [
            {"role": "user", "content": prompt}
        ]

        # Call GPT
        resp = client.chat.completions.create(model="gpt-4", messages=messages)
        content = resp.choices[0].message.content
        match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        sql = match.group(1).strip() if match else content.strip()

        # Display SQL
        st.chat_message("assistant").markdown("```sql\n" + sql + "\n```")

        # Execute and show
        try:
            df_res = con.execute(sql).df()
            if df_res.empty:
                st.warning("No results found.")
            else:
                df_clean = df_res.reset_index(drop=True)
                for c in df_clean.select_dtypes(include=['datetime64[ns]']):
                    df_clean[c] = df_clean[c].dt.strftime('%Y-%m')
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(df_clean.columns)),
                    cells=dict(values=[df_clean[col] for col in df_clean.columns])
                )])
                st.chat_message("assistant").plotly_chart(fig, use_container_width=True)
                st.session_state.chat_history.append(("assistant", df_clean))
        except Exception as e:
            st.chat_message("assistant").write(f"SQL Error: {e}")

    # Render history
    for role, msg in st.session_state.chat_history:
        if role == "assistant" and isinstance(msg, pd.DataFrame):
            st.dataframe(msg)
        else:
            st.write(msg)
