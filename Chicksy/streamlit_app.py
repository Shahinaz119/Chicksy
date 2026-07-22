import streamlit as st
from prompting import generate_answer

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="🐥 Chicksy",
    page_icon="🐥",
    layout="wide",
)

# ==========================
# Load CSS
# ==========================
from pathlib import Path

CSS_PATH = Path(__file__).parent / "assets" / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found: {CSS_PATH}")

# ==========================
# Sidebar
# ==========================
with st.sidebar:

    st.markdown("# 🐥 Chicksy")

    st.markdown(
        """
AI-powered Poultry Farm Assistant

---



---

### 📚 Knowledge Base

Aviagen Ross Broiler Handbook

---

Made with ❤️
"""
    )

# ==========================
# Hero
# ==========================

st.markdown(
"""
<h1 style='text-align:center;'>🐥 Chicksy</h1>
<p style='text-align:center;font-size:22px;color:gray;'>
AI-powered Poultry Farm Assistant
</p>
<p style='text-align:center;font-size:17px;'>
Ask anything about broiler management and receive intelligent answers
from the official handbook.
</p>
""",
unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================
# Feature Cards
# ==========================



st.markdown("---")

# ==========================
# Chat
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question=st.chat_input("🐣 Ask Chicksy anything...")

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🐥 Chicksy is thinking..."):

            result=generate_answer(question)

            answer=result["answer"]

            st.markdown(answer)

            st.markdown("---")

            answer = result["answer"]

    

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )



st.markdown(
"""
<div style='text-align:center;color:gray;'>

🐥 Chicksy

Built with ❤️ using

Sentence Transformers • OpenRouter • Streamlit

</div>
""",
unsafe_allow_html=True
)
