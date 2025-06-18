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

# Load OpenAI API key from environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Smart Adhoc Agent", layout="wide")

# Constants
MAX_HISTORY = 10  # max chat entries to keep

# Cache DuckDB connection
def get_connection():
    return duckdb.connect()

# Cache file loading and preprocessing
a@st.cache_data(show_spinner=False)
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
        except:
            pass
    # Convert date-like columns
    def is_date(s):
        parsed = pd.to_datetime(s, errors='coerce')
        return parsed.notnull().sum() / len(parsed) > 0.5
    for name, df in tables.items():
        for col in df.columns:
            if df[col].dtype == object and is_date(df[col]):
                tables[name][col] = pd.to_datetime(df[col], errors='coerce')
    return tables

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chart_history = []

st.title("🤖 Smart Adhoc Agent")

# Sidebar inputs
with st.sidebar:
    st.header("📁 Upload Data")
    file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])
    sheet_url = st.text_input("Paste public Google Sheets URL")
    profiling = st.checkbox("Enable profiling & merge options")

# Load data
tables = load_and_prepare(file, sheet_url)
con = get_connection()

# Profiling and merge controls
if profiling and tables:
    if st.button("Show Table Profiling"):
        for n, df in tables.items():
            st.subheader(f"Profiling: {n}")
            st.dataframe(df.describe(include='all').transpose())
    if len(tables) > 1 and st.button("Auto-merge Tables on Common Columns"):
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
    first = next(iter(tables))
    st.subheader(f"Preview: {first}")
    st.dataframe(tables[first].head())

# Chat input
if tables:
    user_input = st.chat_input("Ask a question about your data")
    if user_input:
        # Log query
        with open('user_logs.csv','a') as f:
            f.write(f"{datetime.now()}\t{user_input}\n")
        # Append history and trim
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
        st.chat_message("user").write(user_input)
        # Build prompt
        schema = "\n".join([f"{n}: {', '.join(df.columns)}" for n, df in tables.items()])
        prompt = f"""
You are a SQL assistant for DuckDB.
Tables and columns:
{schema}
Important:
- Date columns are DATE/TIMESTAMP; use EXTRACT or YEAR().
- Output ONLY SQL query; no extra text.
Question:
{user_input}
"""
        # Chat completion
        messages = [{"role":"system","content":"You are a SQL expert."}]
        for r, m in st.session_state.chat_history:
            messages.append({"role":r,"content": m if isinstance(m,str) else '<table>'})
        messages.append({"role":"user","content":prompt})
        resp = client.chat.completions.create(model="gpt-4", messages=messages)
        text = resp.choices[0].message.content
        m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL|re.IGNORECASE)
        sql = m.group(1).strip() if m else text.strip()
        # Display and execute
        st.chat_message("assistant").markdown(f"```sql
{sql}
```")
        try:
            df = con.execute(sql).df()
            if df.empty:
                st.warning("No results found.")
            else:
                df = df.reset_index(drop=True)
                for c in df.select_dtypes(include=['datetime64']):
                    df[c] = df[c].dt.strftime('%Y-%m')
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns)),
                    cells=dict(values=[df[col] for col in df.columns])
                )])
                st.chat_message("assistant").plotly_chart(fig, use_container_width=True)
                st.session_state.chat_history.append(("assistant", df))
        except Exception as e:
            st.chat_message("assistant").write(f"SQL Error: {e}")
    # Render history
    for role, msg in st.session_state.chat_history:
        if role=='assistant' and isinstance(msg,pd.DataFrame):
            st.dataframe(msg)
        elif role=='assistant':
            st.write(msg)
        else:
            st.write(msg)
else:
    st.info("Upload a file or enter a Google Sheet URL to start.")
