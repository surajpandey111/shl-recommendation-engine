2. **Access the App**:
- Visit `http://localhost:8501` in your browser.
- Use filters, sample questions, or enter a custom query.

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
