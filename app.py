import streamlit as st
import pandas as pd
import io
import requests

st.set_page_config(page_title="Smart Data Agent", layout="wide")

# --- SESSION SETUP ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# --- SIDEBAR: Upload + Theme Toggle ---
with st.sidebar:
    st.header("📁 Upload File")
    file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])
    sheet_url = st.text_input("Paste a public Google Sheets link")

    st.divider()
    st.subheader("🎨 Theme")
    theme_toggle = st.checkbox("🌗 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = theme_toggle
    selected_theme = "dark" if theme_toggle else "light"

# --- THEME STYLING ---
def apply_theme(theme):
    if theme == "dark":
        st.markdown("""
            <style>
            html, body, [class*="css"]  {
                background-color: #0e1117 !important;
                color: #ffffff !important;
            }
            .stChatMessage {
                background-color: #1c1f26 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            html, body, [class*="css"]  {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            </style>
        """, unsafe_allow_html=True)

apply_theme(selected_theme)

# --- PAGE TITLE ---
st.markdown("<h1 style='text-align: center;'>🤖 Smart Data Agent</h1>", unsafe_allow_html=True)

# --- FILE LOADING ---
df = None

def is_valid_url(url):
    try:
        r = requests.head(url, allow_redirects=True)
        return r.status_code == 200
    except:
        return False

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        st.chat_message("assistant").write("✅ CSV loaded successfully!")

    elif file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(file)
        sheets = xls.sheet_names
        st.chat_message("assistant").write(f"📄 Uploaded Excel with **{len(sheets)}** sheet(s): {', '.join(sheets)}")

        common_cols = None
        sheet_data = {}

        # Load each sheet individually
        for sheet in sheets:
            temp_df = xls.parse(sheet)
            sheet_data[sheet] = temp_df
            if common_cols is None:
                common_cols = set(temp_df.columns)
            else:
                common_cols &= set(temp_df.columns)

        # User selects which sheet to preview
        selected_sheet = st.selectbox("🗂 Select a sheet to preview", options=sheets)
        df = sheet_data[selected_sheet]
        st.chat_message("assistant").write(f"📊 Preview of **{selected_sheet}**:")
        st.dataframe(df.head())

        # Try merging only if common columns exist
        if len(sheet_data) > 1:
            if common_cols:
                st.chat_message("assistant").write(
                    f"🔗 Merging sheets on common columns: {', '.join(common_cols)}"
                )
                try:
                    merged_df = pd.concat(
                        [df[list(common_cols)] for df in sheet_data.values()],
                        ignore_index=True
                    )
                    df = merged_df  # This will be used for GPT later
                    st.chat_message("assistant").write("✅ Sheets merged successfully!")
                except Exception as e:
                    st.chat_message("assistant").write(f"⚠️ Merge failed: {e}")
            else:
                st.chat_message("assistant").write("❌ No common columns found. Skipping merge.")

elif sheet_url:
    if is_valid_url(sheet_url):
        try:
            csv_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
            data = requests.get(csv_url).content
            df = pd.read_csv(io.StringIO(data.decode("utf-8")))
            st.chat_message("assistant").write("✅ Google Sheet loaded successfully!")
        except Exception as e:
            st.chat_message("assistant").write(f"❌ Could not read: {e}")
    else:
        st.chat_message("assistant").write("🔒 This link appears private or invalid.")

# --- PREVIEW + SUGGESTIONS ---
if df is not None:
    # st.chat_message("assistant").write("📊 Here's a preview of your uploaded data:")
    # st.dataframe(df.head())

    st.chat_message("assistant").write("💡 Suggested questions you can ask:")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    suggestions = []

    if numeric_cols and text_cols:
        suggestions.append(f"📊 What is the average {numeric_cols[0]} by {text_cols[0]}?")
        suggestions.append(f"📈 Show the trend of {numeric_cols[0]} over time")
        suggestions.append(f"📌 Top 5 {text_cols[0]} by average {numeric_cols[0]}")
        suggestions.append(f"📍 Total {numeric_cols[0]} for each {text_cols[0]}")

    if "date" in "".join(df.columns).lower():
        suggestions.append("📆 Records created in the last 30 days?")
        suggestions.append("📅 Monthly breakdown of records")

    if len(df.columns) >= 2:
        suggestions.append(f"🔎 Most common combos of {df.columns[0]} and {df.columns[1]}")

    for q in suggestions:
        st.chat_message("assistant").markdown(f"- {q}")

# --- CHAT INPUT BOX ---
if df is not None:
    user_input = st.chat_input("Ask your question about the data...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        response = "🧠 GPT will respond here tomorrow..."
        st.session_state.chat_history.append(("assistant", response))
        st.chat_message("assistant").write(response)
else:
    st.chat_input("Upload a file first to enable chat.")
