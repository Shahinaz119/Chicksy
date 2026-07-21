import streamlit as st
from prompting import generate_answer

# ==================================
# Page Config
# ==================================

st.set_page_config(
    page_title="Chicksy",
    page_icon="🐥",
    layout="wide"
)

# ==================================
# Custom CSS
# ==================================

st.markdown("""
<style>

.main {
    background-color: #f8faf8;
}

h1 {
    color:#2E7D32;
    text-align:center;
}

.stButton>button{
    width:100%;
    background:#43A047;
    color:white;
    border-radius:12px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#2E7D32;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# Sidebar
# ==================================

with st.sidebar:

    st.title("🐥 Chicksy")

    st.markdown("---")

    st.write("### AI Broiler Farm Assistant")

    st.write("""
This chatbot answers questions from the
ROSS Broiler Management Handbook.


""")

    st.markdown("---")

    

# ==================================
# Main Page
# ==================================

st.title("🐥 Chicksy")

st.subheader("Broiler Farm AI Assistant")

st.write(
    "Ask any question about broiler management."
)

question = st.text_input(
    "Enter your question"
)

# ==================================
# Generate Answer
# ==================================

if st.button("🚀 Ask Chicksy"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            result = generate_answer(
                question
            )

        st.success("Answer")

        st.write(result["answer"])

        st.markdown("---")

        st.subheader("📚 Retrieved Chunks")

        for _, row in result["retrieved_chunks"].iterrows():

            with st.expander(
                f"Chunk {row['chunk_id']}"
            ):

                st.write(
                    row["document"]
                )