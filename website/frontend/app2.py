import streamlit as st  # pyright: ignore[reportMissingImports]
import requests

st.set_page_config(page_title="JD Architect", layout="wide")

st.title("📄 Professional JD Architect")
st.caption("Custom Fine-tuned Gemma-2B Backend")

if "current_jd" not in st.session_state:
    st.session_state.current_jd = None

# Sidebar
with st.sidebar:
    if st.button("Reset / New Project"):
        st.session_state.current_jd = None
        st.rerun()

label = "Edit Instruction:" if st.session_state.current_jd else "Enter Raw Requirements:"
user_query = st.text_area(label, height=150)

if st.button("Submit"):
    if user_query:
        with st.spinner("AI is generating..."):
            try:
                payload = {
                    "user_input": user_query,
                    "current_state": st.session_state.current_jd
                }

                # ✅ FIXED PORT
                response = requests.post("http://localhost:8005/process", json=payload)

                if response.status_code == 200:
                    result = response.json()

                    if isinstance(result, dict) and "error" in result:
                        st.error(f"Model Error: {result['error']}")
                        st.expander("Raw Output").text(result.get("raw", ""))
                    else:
                        st.session_state.current_jd = result
                else:
                    st.error(f"Backend Error: {response.text}")

            except Exception as e:
                st.error(f"Connection Error: {e}")

# Display
if st.session_state.current_jd:
    left, right = st.columns(2)

    with left:
        st.subheader("Raw JSON")
        st.json(st.session_state.current_jd)

    with right:
        st.subheader("Preview")
        jd = st.session_state.current_jd

        st.write(f"### {jd.get('job_title', 'JD')}")
        st.info(f"📍 {jd.get('location', 'N/A')} | 🏢 {jd.get('industry', 'N/A')}")

        st.markdown("**Responsibilities:**")
        for item in jd.get('responsibilities', []):
            st.markdown(f"- {item}")