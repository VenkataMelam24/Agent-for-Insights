import streamlit as st
import pandas as pd
import duckdb
import io
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import plotly.graph_objects as go
from datetime import datetime  # ⬅️ NEW: for timestamp logging

# Load OpenAI API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Smart Adhoc Agent", layout="wide")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# NEW 🔐 Ask user name/email before proceeding
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

if not st.session_state.user_id:
    st.session_state.user_id = st.text_input("🔐 Please enter your name or email to begin:")

if not st.session_state.user_id:
    st.stop()

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

# Register with DuckDB and show preview
if tables:
    con = duckdb.connect()
    for name, df in tables.items():
        try:
            con.register(name, df)
        except Exception as e:
            st.warning(f"❌ Failed to register {name}: {e}")

    # Attempt automatic merge if common columns exist
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

# User chat
if tables:
    user_input = st.chat_input("Ask a question about your data")
    if user_input:
        # NEW ✍️ Log user ID + query + time
        with open("user_logs.csv", "a") as f:
            f.write(f"{st.session_state.user_id},{user_input},{datetime.now()}\n")

        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        try:
            schema_description = "\n".join([
                f"{name} → {', '.join(df.columns)}" for name, df in tables.items()
            ])

            prompt = f"""You are an expert SQL assistant using DuckDB.
Only use the following tables and columns:
{schema_description}

Rules:
- Do not guess column names.
- Use only listed tables/columns.
- If user asks about time trends, prefer grouping by month if date fields are present.
- Output only SQL code without explanation.

Question:
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

            sql_code = response.choices[0].message.content.strip("```sql").strip("```")

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
                    cells=dict(values=[msg[col] for col in msg.columns], align='left')
                )])
                st.chat_message("assistant").plotly_chart(table, use_container_width=True, key=f"history_plot_{hash(str(msg))}")
            else:
                st.chat_message("assistant").write(str(msg))
else:
    st.chat_input("Please upload a file to begin.")
