"""
Streamlit UI for Vietnamese Law RAG System
"""
import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.rag import RAGService
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="Vietnamese Law RAG",
    page_icon="⚖️",
    layout="wide"
)

# Initialize RAG service
@st.cache_resource
def get_rag_service():
    """Initialize and cache RAG service"""
    logger.info("Initializing RAG service for Streamlit")
    return RAGService(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="Law",
        es_host="localhost",
        es_port=9200,
        es_index_name="law_documents",
        top_k=5
    )

# Main UI
def main():
    st.title("⚖️ Vietnamese Law RAG System")
    st.markdown("Hệ thống hỏi đáp về luật giao thông đường bộ Việt Nam")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Display system info
        st.info("""
        **Hệ thống:**
        - Vector DB: Qdrant (semantic search)
        - Keyword DB: Elasticsearch (BM25)
        - Retrieval: EnsembleRetriever
        - Reranker: BGE-reranker-v2-m3
        - LLM: Qwen2.5-7B-Instruct
        """)
        
        # Advanced settings
        with st.expander("🔧 Cài đặt nâng cao"):
            top_k = st.slider("Số documents retrieve", 3, 10, 5)
            rerank_top_k = st.slider("Số documents sau rerank", 1, 5, 3)
            show_sources = st.checkbox("Hiển thị nguồn", value=True)
            show_debug = st.checkbox("Debug mode", value=False)
        
        # Clear chat button
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and show_sources:
                with st.expander("📚 Nguồn tham khảo"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {source['title']}**")
                        st.markdown(f"> {source['content'][:200]}...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi của bạn về luật giao thông..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
                try:
                    # Get RAG service
                    rag_service = get_rag_service()
                    
                    # Generate response (using sync version)
                    result = rag_service.generate_response_sync(prompt)
                    
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Prepare sources for display
                    sources = []
                    if "source_documents" in result:
                        for doc in result["source_documents"]:
                            sources.append({
                                "title": doc.metadata.get("title", "No title"),
                                "content": doc.page_content
                            })
                    
                    # Show sources if enabled
                    if sources and show_sources:
                        with st.expander("📚 Nguồn tham khảo"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source['title']}**")
                                st.markdown(f"> {source['content'][:200]}...")
                                if show_debug:
                                    st.code(source['content'])
                                st.markdown("---")
                    
                    # Debug info
                    if show_debug:
                        with st.expander("🔍 Debug Info"):
                            st.json({
                                "query": prompt,
                                "num_sources": len(sources),
                                "answer_length": len(result["answer"])
                            })
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Lỗi: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error generating response: {e}", exc_info=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
