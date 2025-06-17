import streamlit as st
import pandas as pd
import duckdb
import io
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests

# Load OpenAI API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Smart Data Agent", layout="wide")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Smart Data Agent")

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

    first_key = next(iter(tables))
    df_preview = tables[first_key]
    st.subheader("📊 Preview of first table")
    st.dataframe(df_preview.head())

# User chat
if tables:
    user_input = st.chat_input("Ask a question about your data")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        try:
            schema_description = "\n".join([
                f"{name} → {', '.join(df.columns)}" for name, df in tables.items()
            ])

            prompt = f"""You are a SQL analyst using DuckDB.
Available tables and their columns:
{schema_description}

Write a valid SQL query using DuckDB syntax to answer:
{user_input}
Only return the SQL query, nothing else."""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a SQL expert helping a user with their data."},
                    {"role": "user", "content": prompt}
                ]
            )

            sql_code = response.choices[0].message.content.strip("```sql").strip("```")

            st.chat_message("assistant").markdown(f"💡 SQL Query:\n```sql\n{sql_code}\n```")

            result_df = con.sql(sql_code).df()
            st.chat_message("assistant").write(result_df)
            st.session_state.chat_history.append(("assistant", result_df))

        except Exception as e:
            st.chat_message("assistant").write(f"⚠️ GPT or SQL Error: {e}")

    # Show chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            if isinstance(msg, pd.DataFrame):
                st.chat_message("assistant").write(msg)
            else:
                st.chat_message("assistant").write(str(msg))
else:
    st.chat_input("Please upload a file to begin.")
