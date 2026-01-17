"""
Streamlit Frontend - Giao diện chatbot cho hệ thống RAG
"""
import streamlit as st
import httpx
import asyncio
from typing import List, Dict

# Cấu hình trang
st.set_page_config(
    page_title="Luật Giao thông Việt Nam - Chatbot",
    page_icon="🚦",
    layout="wide"
)

# URL của backend API
API_URL = "http://localhost:8000"


async def query_api(question: str, top_k: int = 20, top_n: int = 5) -> Dict:
    """
    Gọi API backend để lấy câu trả lời
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{API_URL}/query",
            json={
                "question": question,
                "top_k": top_k,
                "top_n": top_n
            }
        )
        response.raise_for_status()
        return response.json()


def display_message(role: str, content: str):
    """
    Hiển thị message trong chat interface
    """
    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").write(content)


def main():
    """Main application"""
    
    # Header
    st.title("🚦 Tư vấn Luật Giao thông Việt Nam")
    st.markdown("""
    Hệ thống hỏi đáp thông minh về Luật Giao thông Việt Nam  
    *Powered by Hybrid Ensemble Agentic RAG - DeepSeek R1*
    """)
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        top_k = st.slider(
            "Số lượng documents tìm kiếm (Top K)",
            min_value=5,
            max_value=50,
            value=20,
            help="Số lượng documents lấy từ mỗi search engine"
        )
        
        top_n = st.slider(
            "Số lượng contexts cho LLM (Top N)",
            min_value=1,
            max_value=10,
            value=5,
            help="Số lượng contexts sau rerank để đưa vào LLM"
        )
        
        show_reasoning = st.checkbox(
            "Hiển thị quá trình suy luận",
            value=False,
            help="Hiển thị phần <thinking> của model"
        )
        
        show_contexts = st.checkbox(
            "Hiển thị contexts",
            value=True,
            help="Hiển thị các đoạn văn bản tham khảo"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Ví dụ câu hỏi")
        st.markdown("""
        - Vượt đèn đỏ xe máy phạt bao nhiêu?
        - Không đội mũ bảo hiểm bị phạt thế nào?
        - Điều khiển xe khi say rượu bị xử phạt ra sao?
        - Tốc độ tối đa trong khu dân cư là bao nhiêu?
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi của bạn về luật giao thông..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)
        
        # Show loading spinner
        with st.spinner("🤔 Đang tìm kiếm và phân tích..."):
            try:
                # Call API
                result = asyncio.run(query_api(prompt, top_k, top_n))
                
                # Display answer
                answer = result.get("answer", "Không có câu trả lời")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                display_message("assistant", answer)
                
                # Display reasoning if enabled
                if show_reasoning and result.get("reasoning"):
                    with st.expander("🧠 Quá trình suy luận"):
                        st.markdown(result["reasoning"])
                
                # Display contexts if enabled
                if show_contexts and result.get("contexts"):
                    with st.expander(f"📚 Tài liệu tham khảo ({len(result['contexts'])} documents)"):
                        for i, ctx in enumerate(result["contexts"]):
                            st.markdown(f"**Document {i+1}** (Score: {ctx.get('score', 0):.4f})")
                            st.info(ctx.get("content", ""))
                            st.markdown("---")
            
            except Exception as e:
                error_msg = f"❌ Lỗi: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.sidebar.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 💡 Về hệ thống
    - **Vector DB**: Qdrant
    - **Keyword Search**: Elasticsearch  
    - **Fusion**: RRF Algorithm
    - **Reranker**: BGE-Reranker-v2-m3
    - **LLM**: DeepSeek-R1-7B
    """)


if __name__ == "__main__":
    main()
