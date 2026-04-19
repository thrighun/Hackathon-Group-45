import streamlit as st
import requests
import json

BACKEND_URL = "http://127.0.0.1:8006/chat"

st.title("💼 JD Generator + Editor")

if "jd" not in st.session_state:
    st.session_state.jd = None

# INPUT
user_input = st.text_area("Enter Job Description or Instruction")

col1, col2 = st.columns(2)

# SEND
if col1.button("Send"):

    try:
        response = requests.post(
            BACKEND_URL,
            json={"text": user_input}
        )

        result = response.json()

        if "json_data" in result:
            st.session_state.jd = result["json_data"]
            st.json(result["json_data"])
        else:
            st.write(result)

    except Exception as e:
        st.error(str(e))

# RESET
if col2.button("Reset"):

    requests.post(
        BACKEND_URL,
        json={"text": "", "reset": True}
    )

    st.session_state.jd = None
    st.success("Reset done")

# DOWNLOAD
if st.session_state.jd:
    st.download_button(
        "Download JSON",
        data=json.dumps(st.session_state.jd, indent=2),
        file_name="jd.json"
    )