import os
import io
import re
from datetime import datetime

import streamlit as st
import pandas as pd
import duckdb
import requests
from dotenv import load_dotenv
from openai import OpenAI
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# Config & Client
# ──────────────────────────────────────────────
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Smart Adhoc Agent", layout="wide")
st.title("🤖 Smart Adhoc Agent")

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_tables(uploader, sheet_url):
    tables = {}
    if uploader:
        fn = uploader.name.lower()
        if fn.endswith(".csv"):
            tables["data"] = pd.read_csv(uploader)
        elif fn.endswith(".xlsx"):
            xls = pd.ExcelFile(uploader)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                key = re.sub(r"[^\w]+", "_", sheet.strip().lower())
                tables[key] = df
    elif sheet_url and "docs.google.com" in sheet_url:
        csv_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
        resp = requests.get(csv_url)
        resp.raise_for_status()
        tables["data"] = pd.read_csv(io.StringIO(resp.text))
    return tables

def convert_dates(df):
    for c in df.select_dtypes("object"):
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.5:
            df[c] = parsed
    return df

def prepare_duckdb(tables):
    con = duckdb.connect()
    for name, df in tables.items():
        tables[name] = convert_dates(df)
        con.register(name, tables[name])
    if len(tables) > 1:
        common = set.intersection(*(set(df.columns) for df in tables.values()))
        if common:
            joins = " ".join(
                f"JOIN {t} USING ({','.join(common)})"
                for t in list(tables)[1:]
            )
            sql = f"SELECT * FROM {list(tables)[0]} {joins}"
            merged = con.execute(sql).df()
            con.register("merged_data", merged)
            tables["merged_data"] = merged
    return con, tables

def make_prompt(tables, q):
    schema = "\n".join(f"{n}: {', '.join(df.columns)}" for n, df in tables.items())
    return f"""You are a DuckDB SQL expert. Only use these tables/columns:
{schema}

Use DuckDB date functions. Output ONLY valid SQL.

QUESTION:
{q}
"""

def extract_sql(resp_text):
    m = re.search(r"```sql\s*(.+?)```", resp_text, re.DOTALL)
    return m.group(1).strip() if m else resp_text.strip()

def run_query(con, sql):
    df = con.execute(sql).df().reset_index(drop=True)
    for c in df.select_dtypes("datetime64[ns]"):
        df[c] = df[c].dt.strftime("%Y-%m")
    return df

def display_table(df):
    """Return a styled Plotly table figure for embedding in chat or main view."""
    rows, cols = df.shape
    width = min(cols * 100 + 200, 1200)
    height = min(rows * 30 + 50, 800)
    colors = ["white" if i % 2 == 0 else "lightgrey" for i in range(rows)]
    fig = go.Figure(
        go.Table(
            header=dict(values=list(df.columns), fill_color="darkslategray", font=dict(color="white", size=14)),
            cells=dict(values=[df[c] for c in df.columns], fill_color=[colors], align="left")
        )
    )
    fig.update_layout(width=width, height=height, margin=dict(t=10,b=10,l=10,r=10))
    return fig

# ──────────────────────────────────────────────
# Sidebar: Data Load
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded = st.file_uploader("Excel or CSV", ["csv","xlsx"] )
    gsheet   = st.text_input("Google Sheet URL")

# ──────────────────────────────────────────────
# Load Data & Notify
# ──────────────────────────────────────────────
tables = load_tables(uploaded, gsheet)
if not tables:
    st.warning("Please upload a file or paste a Google Sheet URL.")
    st.stop()
else:
    st.success(f"✅ Data loaded successfully! Tables: {', '.join(tables.keys())}")

# ──────────────────────────────────────────────
# Prepare DB
# ──────────────────────────────────────────────
con, tables = prepare_duckdb(tables)

# ──────────────────────────────────────────────
# Chat Input & Response
# ──────────────────────────────────────────────
q = st.chat_input("Ask a question about your data")
if q:
    with open("user_logs.csv","a") as log:
        log.write(f"{datetime.now()},{q}\n")
    st.chat_message("user").write(q)
    st.session_state.chat_history.append(("user", q))

    prompt = make_prompt(tables, q)
    msgs = [
        {"role":"system","content":"You output DuckDB SQL only."},
        {"role":"user","content":prompt}
    ]
    resp = openai.chat.completions.create(model="gpt-4", messages=msgs)
    sql = extract_sql(resp.choices[0].message.content)

    # run & display only table
    try:
        df = run_query(con, sql)
        st.session_state.chat_history.append(("assistant", df))
        fig = display_table(df)
        idx = len(st.session_state.chat_history)
        st.chat_message("assistant").plotly_chart(fig, use_container_width=False, key=f"table_{idx}")
    except Exception as e:
        st.chat_message("assistant").write(f"⚠️ SQL Error: {e}")

# ──────────────────────────────────────────────
# Chat History (optional)
# ──────────────────────────────────────────────
with st.expander("🕘 Chat History", expanded=False):
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            if isinstance(msg, pd.DataFrame):
                fig = display_table(msg)
                loop_idx = hash(str(msg))
                st.plotly_chart(fig, use_container_width=False, key=f"hist_table_{loop_idx}")
            else:
                st.markdown(f"**Agent:** {msg}")



