# pyright: reportMissingImports=false
import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(
    page_title="AI Job Parser",
    page_icon="💼",
    layout="wide"
)

# =========================
# SESSION STATE
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8006/parse"

# =========================
# HELPER
# =========================
def push_chat(user_text, status, output):
    st.session_state.chat.insert(0, {
        "time": datetime.now().strftime("%H:%M"),
        "input": user_text,
        "status": status,
        "output": output
    })

# =========================
# UI
# =========================
st.title("💼 AI Job Parser")
st.caption("Gemma-2 Fine-tuned Backend")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    st.text_input("Backend URL", key="backend_url")

    if st.button("🗑 Clear History"):
        st.session_state.chat = []
        st.rerun()

# Input
user_input = st.text_area(
    "Paste Job Description",
    height=200,
    placeholder="Paste full job description here..."
)

# Button (ONLY ONE NOW)
generate_btn = st.button("⚡ Generate JSON", use_container_width=True)

# =========================
# GENERATE
# =========================
if generate_btn:
    if not user_input.strip():
        st.warning("Please enter job description")
    else:
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    st.session_state.backend_url,
                    json={"description": user_input},
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json().get("json_data", {})

                    if isinstance(result, dict) and "error" not in result:
                        st.success("✅ Parsed successfully")
                        st.json(result, expanded=True)

                        push_chat(user_input, "success", result)

                        st.download_button(
                            "⬇ Download JSON",
                            data=json.dumps(result, indent=4),
                            file_name="job.json",
                            mime="application/json"
                        )

                    else:
                        st.error("⚠ Model returned invalid JSON")
                        st.code(result)

                        push_chat(user_input, "error", result)

                else:
                    st.error(f"Backend Error: {response.text}")
                    push_chat(user_input, "error", response.text)

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend (check port & server)")
            except requests.exceptions.Timeout:
                st.error("⏳ Request timeout (model slow)")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# =========================
# HISTORY
# =========================
st.subheader("📜 History")

if not st.session_state.chat:
    st.info("No history yet")
else:
    for item in st.session_state.chat[:10]:
        st.markdown(f"**[{item['time']}] {item['status'].upper()}**")
        st.text(item["input"][:100])

        with st.expander("View Output"):
            st.json(item["output"])

        st.markdown("---")