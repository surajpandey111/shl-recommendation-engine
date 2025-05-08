# SHL Assessment Recommendation Engine (Advanced RAG Approach)

This repository contains an advanced implementation of the SHL Assessment Recommendation Engine, developed for the SHL Research Intern application. The project leverages Retrieval-Augmented Generation (RAG) with LangChain, FAISS, and HuggingFace embeddings to recommend SHL assessments using semantic search. You can explore the live demo of the web app at [https://shl-recommendation-engine-suraj.streamlit.app/](https://shl-recommendation-engine-suraj.streamlit.app/).

## Project Overview

The SHL Assessment Recommendation Engine assists users in finding suitable SHL assessments by processing queries like “Which assessment is suitable for an entry-level customer service role?”. It employs an advanced RAG approach with semantic search to retrieve relevant assessments from a product catalog and generate concise recommendations.

- **GitHub Repository**: [https://github.com/surajpandey111/shl-recommendation-engine](https://github.com/surajpandey111/shl-recommendation-engine)

### Features
- **Semantic Retrieval**: Utilizes LangChain, FAISS, and HuggingFace embeddings (`all-MiniLM-L6-v2`) for vector-based semantic search, enabling accurate retrieval for complex queries.
- **Recommendation Generation**: Produces concise, relevant recommendations based on retrieved assessments.
- **User Interface**: A Streamlit web app (`shl_engine.py`) with:
  - Filters for Job Level and Test Duration.
  - Sample questions and custom query input.
  - Clean display of recommendations.
- **Error Handling**: Robust handling for CSV loading, encoding issues, and API failures with user-friendly messages.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Gemini API key (for recommendation generation)
- `shl_product_catalog.csv` file in the project directory

### Installation
1. **Clone the Repository**:
git clone https://github.com/surajpandey111/shl-recommendation-engine.git
cd shl-recommendation-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Required packages: `streamlit`, `pandas`, `langchain-core`, `langchain-community`, `langchain-huggingface`, `faiss-cpu`, `google-generativeai`.

3. **Set Up the Gemini API Key**:
- Set the environment variable:
export GEMINI_API_KEY='your-gemini-api-key'  # On Windows: set GEMINI_API_KEY=your-gemini-api-key
- Or replace `'your-actual-api-key'` in `shl_engine.py` with your key.

4. **Prepare the Product Catalog**:
- Place `shl_product_catalog.csv` in the project directory.
- Required columns: `Product Name`, `Description`, `Job Level`, `Languages`, `Test Duration`, `Test Type`, `Remote Testing`.

### Running the App
1. **Start the Streamlit App**:
streamlit run shl_engine.py
2. **Access the App**:
- Visit `http://localhost:8501` in your browser.
- Use filters, sample questions, or enter a custom query to get recommendations.

## Development Process

This project was developed to explore advanced RAG techniques, focusing on semantic search for improved retrieval accuracy.

1. **Data Preparation**:
- Loaded `shl_product_catalog.csv` using `pandas`, with error handling for file issues, parsing errors, and encoding.
- Converted CSV rows into `Document` objects (LangChain’s `langchain_core.documents`), combining fields for search.

2. **Embedding and Vector Search**:
- Generated embeddings using HuggingFace’s `all-MiniLM-L6-v2` model (`langchain_huggingface`).
- Stored embeddings in a FAISS vector store (`langchain_community.vectorstores`) for similarity search.
- Cached embeddings with `@st.cache_resource` for performance.

3. **Retrieval Logic**:
- Used semantic search to retrieve the top 8 documents via `db.similarity_search_with_score`.
- Applied Job Level and Test Duration filters, with fallback to unfiltered results if needed.
- Dynamically populated filter options from the dataset.

4. **Recommendation Generation**:
- Constructed prompts with retrieved documents and user queries.
- Generated recommendations using the Gemini API (`gemini-2.0-flash`), with error handling for API failures.

5. **Web App Development**:
- Built a Streamlit app (`shl_engine.py`) with a wide layout.
- Designed an intuitive UI with filters, sample questions, custom query input, and a recommendation display.
- Added a footer crediting the SHL Research Intern application.

6. **Deployment**:
- Deployed on Streamlit Community Cloud, linked to this repository.
- Managed dependencies in `requirements.txt` for smooth deployment.

## Submission Details
Submitted for the SHL Research Intern application (late submission). This project demonstrates advanced RAG techniques using LangChain, FAISS, and HuggingFace embeddings for SHL assessment recommendations.

## Limitations
- Requires a Gemini API key and may face rate limits.
- Initial queries on Streamlit Community Cloud may have delays (free tier).
- Relies on a correctly formatted `shl_product_catalog.csv`.

## Future Improvements
- Add filters for Language and Test Type.
- Use a local recommendation model to reduce API dependency.
- Optimize FAISS for larger datasets.

## Author
Built with ❤️ by Suraj Kumar Pandey for the SHL Research Intern Application.
