# ResumeRAG - A Resume-Screening-RAG
##Overview

Resume Screening RAG is an agentic Retrieval-Augmented Generation (RAG) application designed to assist hiring managers in efficiently screening resumes based on job descriptions. The system retrieves and summarizes the most relevant resumes, allowing users to compare candidates directly or obtain top-ranked candidates for a given job description.

###Features

🔍 Semantic Resume Search: Retrieve the most relevant resumes using semantic search.

📑 Summarization: Generate concise summaries of candidates based on job-specific relevance.

⚙️ Comparison Mode: Provide application IDs to compare selected candidates.

🎯 Ranking Mode: Input a job description to get the top-ranked candidates.

🔄 Configurable Retrieval: Choose between:

Standard RAG: Traditional retrieval-augmented generation for fetching resumes.

RAG Fusion: Advanced fusion-based retrieval for improved relevance.

##Technologies Used

LLM Agent: OpenAI API for response generation.

Vector Database: FAISS for semantic search.

Framework: LangChain for retrieval-augmented generation pipeline.

Deployment: Streamlit for interactive UI.

##Installation

git clone [https://github.com/your-repo/resume-screening-rag.git](https://github.com/Senzen18/ResumeRAG.git)
cd resume-screening-rag
pip install -r requirements.txt

##Usage

Run the application:

streamlit run run.py


##Future Enhancements

🔗 Graph-Based Retrieval: Enhance search using knowledge graphs.

🏆 Candidate Ranking Metrics: Improve ranking accuracy.

📊 Dashboard UI: Expand interactive visualization capabilities.

##Contributing

Feel free to contribute by submitting pull requests or opening issues.

License

MIT License
