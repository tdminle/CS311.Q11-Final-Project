"""
Generation Module - Gọi LLM DeepSeek-R1 để generate câu trả lời
"""
import httpx
from src.app.config import get_settings
from src.app.prompts.templates import build_prompt
from typing import List, Dict, Tuple
import re
from langfuse import observep

settings = get_settings()


@observe(as_type="generation", name="generate_answer")
async def generate_answer(query: str, contexts: List[Dict]) -> Tuple[str, str]:
    """
    Generate câu trả lời dựa trên query và contexts sử dụng DeepSeek-R1
    
    Args:
        query: Câu hỏi của người dùng
        contexts: List các context documents đã được rerank
    
    Returns:
        Tuple (answer, reasoning) - Câu trả lời và phần suy luận
    """
    # Build prompt từ template
    prompt = build_prompt(query, contexts)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Gọi API LLM
            response = await client.post(
                settings.LLM_API_URL,
                json={
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024
                },
                headers={"Authorization": f"Bearer {settings.API_KEY}"} if settings.API_KEY else {}
            )
            response.raise_for_status()
            result = response.json()
        
        # Parse response từ LLM
        # Format thường là: {"choices": [{"message": {"content": "..."}}]}
        full_response = result["choices"][0]["message"]["content"]
        
        # Tách reasoning và answer nếu có tag <thinking>
        reasoning = ""
        answer = full_response
        
        # Tìm phần reasoning trong <thinking> tags
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
        if thinking_match:
            reasoning = thinking_match.group(1).strip()
            # Remove thinking part để lấy answer
            answer = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL).strip()
        
        print(f"🤖 Generated answer (length: {len(answer)} chars)")
        
        return answer, reasoning
    
    except Exception as e:
        print(f"❌ Lỗi khi gọi LLM API: {e}")
        return "Xin lỗi, tôi không thể tạo câu trả lời lúc này. Vui lòng thử lại sau.", ""
