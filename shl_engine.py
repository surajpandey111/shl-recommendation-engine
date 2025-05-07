import streamlit as st
import pandas as pd
import csv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import os

st.set_page_config(page_title="SHL Assessment Recommendation Engine", page_icon="🧠", layout="wide")


api_key = os.getenv("GEMINI_API_KEY", "your-actual-api-key")
if api_key == "your-actual-api-key":
    st.error("❌ Error: Gemini API key not set. Please set the 'GEMINI_API_KEY' environment variable or replace 'your-actual-api-key' in the script.")
    st.stop()

genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-2.0-flash")

@st.cache_resource
def load_documents():
    try:
        # Load the CSV file with proper quoting and encoding
        df = pd.read_csv("shl_product_catalog.csv", quoting=csv.QUOTE_ALL, escapechar='\\', encoding='utf-8')
    except FileNotFoundError:
        st.error("❌ Error: 'shl_product_catalog.csv' not found. Please ensure the file is in the project directory.")
        return []
    except pd.errors.EmptyDataError:
        st.error("❌ Error: 'shl_product_catalog.csv' is empty. Please provide a valid CSV file.")
        return []
    except pd.errors.ParserError:
        st.error("❌ Error: Failed to parse 'shl_product_catalog.csv'. Ensure the file is a valid CSV with the correct format.")
        return []
    except UnicodeDecodeError:
        st.error("❌ Error: Unable to read 'shl_product_catalog.csv' due to encoding issues. Ensure the file is saved with UTF-8 encoding.")
        return []
    
    # Define required columns
    required_columns = ['Product Name', 'Description', 'Job Level', 'Languages', 'Test Duration', 'Test Type', 'Remote Testing']
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ Error: CSV must contain all required columns: {required_columns}")
        return []
    
    # Convert each row into a document by combining relevant fields
    documents = []
    for index, row in df.iterrows():
        # Handle NaN or empty values
        row = row.fillna('Not Specified')
        doc_content = (
            f"Product Name: {row['Product Name']}\n"
            f"Description: {row['Description']}\n"
            f"Job Level: {row['Job Level']}\n"
            f"Languages: {row['Languages']}\n"
            f"Test Duration: {row['Test Duration']}\n"
            f"Test Type: {row['Test Type']}\n"
            f"Remote Testing: {row['Remote Testing']}"
        )
        documents.append(Document(page_content=doc_content, metadata={"index": index}))
    
    return documents

@st.cache_resource
def embed_documents(_docs):
    if not _docs:
        st.error("❌ Error: No documents loaded. Cannot create FAISS index. Please check your CSV file.")
        return None
    try:
        texts = [doc.page_content for doc in _docs]
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        db = FAISS.from_texts(texts, embedding=embedding_model, metadatas=[{"text": text} for text in texts])
        return db
    except Exception as e:
        st.error(f"❌ Error: Failed to create FAISS index. This might be due to missing dependencies or network issues. Error: {e}")
        return None

# Initialize documents and vector store
docs = load_documents()
db = embed_documents(docs)

if db is None:
    st.stop()

# Header
st.title("🧠 SHL Assessment Recommendation Engine")
st.markdown("Find the perfect SHL assessment for your hiring needs! Filter by job level and test duration, or ask a specific question.")

# Filters
st.subheader("🔍 Filter Assessments")
col1, col2 = st.columns(2)

# Extract job levels safely
job_levels = set()
for doc in docs:
    split_content = doc.page_content.split("Job Level: ")
    if len(split_content) > 1:
        levels_part = split_content[1].split("\n")[0]
        levels = levels_part.split(";")
        for level in levels:
            if level.strip():
                job_levels.add(level.strip())

job_levels = ["All"] + sorted(job_levels)

# Extract test durations safely
test_durations = set()
for doc in docs:
    split_content = doc.page_content.split("Test Duration: ")
    if len(split_content) > 1:
        duration = split_content[1].split("\n")[0].strip()
        if duration.isdigit():
            test_durations.add(duration)

test_durations = ["All"] + sorted(test_durations, key=int)

with col1:
    selected_job_level = st.selectbox("Job Level", job_levels, index=0)
with col2:
    selected_test_duration = st.selectbox("Test Duration (minutes)", test_durations, index=0)

# Sample Questions
st.subheader("💡 Try a Sample Question")
sample_questions = [
    "Which assessment is suitable for an entry-level customer service role?",
    "Recommend an assessment for a mid-level manager in the insurance industry.",
    "What’s the best assessment for a senior sales professional?"
]
col3, col4, col5 = st.columns(3)

# Initialize session state for sample question
if 'selected_sample_question' not in st.session_state:
    st.session_state.selected_sample_question = ""

# Handle sample question buttons
with col3:
    if st.button(sample_questions[0]):
        st.session_state.selected_sample_question = sample_questions[0]
with col4:
    if st.button(sample_questions[1]):
        st.session_state.selected_sample_question = sample_questions[1]
with col5:
    if st.button(sample_questions[2]):
        st.session_state.selected_sample_question = sample_questions[2]

# Text input with controlled value
user_input = st.text_input(
    "Or ask your own question (e.g., 'Which assessment is suitable for an entry-level customer service role?'):",
    value=st.session_state.selected_sample_question,
    key="user_input"
)

# Process User Input
if not user_input:
    st.info("ℹ️ Please enter a query or select a sample question to get a recommendation.")
else:
    user_input_lower = user_input.lower().strip()
    if any(phrase in user_input_lower for phrase in ["who are you", "what are you", "who is this", "what is this"]):
        st.markdown("### 🧠 Answer:\nI am the SHL Assessment Recommendation Engine, here to help you find the best SHL assessments for specific roles and job levels!")
    else:
        # Get context from FAISS
        search_results = db.similarity_search_with_score(user_input, k=8)
        
        # Filter results based on user-selected filters
        filtered_results = []
        for doc, score in search_results:
            job_level_match = True
            duration_match = True
            
            # Check Job Level
            if selected_job_level != "All":
                split_content = doc.page_content.split("Job Level: ")
                if len(split_content) > 1:
                    levels = split_content[1].split("\n")[0].split(";")
                    job_level_match = selected_job_level in [level.strip() for level in levels]
                else:
                    job_level_match = False
            
            # Check Test Duration
            if selected_test_duration != "All":
                split_content = doc.page_content.split("Test Duration: ")
                if len(split_content) > 1:
                    duration = split_content[1].split("\n")[0].strip()
                    duration_match = duration == selected_test_duration
                else:
                    duration_match = False
            
            if job_level_match and duration_match:
                filtered_results.append((doc, score))
        
        if not filtered_results:
            st.warning("⚠️ No assessments match the selected filters. Showing results without filters.")
            filtered_results = search_results[:8]
        
        # Extract context from filtered results
        context = "\n".join([doc.page_content for doc, _ in filtered_results])

        # Enhanced prompt for better recommendations
        prompt = (
            f"You are an SHL Assessment Recommendation Engine. Use the following context to recommend the most suitable SHL assessments based on the user's query. "
            f"Prioritize the assessment that best matches the query in terms of job level, role, and industry. If multiple assessments are relevant, provide a ranked list (up to 3) with the product name, description, job level, test duration, and a brief reason for the match. "
            f"If no suitable assessment is found, suggest the closest match and explain why, or indicate that no match was found.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_input}"
        )

        try:
            with st.spinner("Generating recommendation..."):
                response = gemini.generate_content(prompt)
                st.markdown("### 🧠 Recommendation:")
                st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Gemini API Error: {e}. Please check your API key or try again later. If the issue persists, ensure your API key is valid and you have not exceeded your usage limits.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Suraj Kumar Pandey for SHL Research Intern Application")