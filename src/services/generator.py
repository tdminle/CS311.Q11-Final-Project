"""
Generator Service for generating responses using LLM.
"""
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from src.models.llm import chat_model
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# System prompt for RAG
SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh và hữu ích.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp.
Nếu thông tin không đủ để trả lời, hãy nói rõ điều đó.
Trả lời bằng tiếng Việt một cách ngắn gọn, chính xác và dễ hiểu.
Không bịa đặt thông tin không có trong ngữ cảnh."""


class GeneratorService:
    """Service for generating responses using LLM."""
    
    def __init__(self, system_prompt: str = None):
        """
        Initialize Generator Service.
        
        Args:
            system_prompt: Custom system prompt (uses default if None)
        """
        logger.info("Initializing GeneratorService")
        
        self.llm = chat_model
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        
        logger.info("✅ GeneratorService initialized")
    
    def _create_messages(self, question: str, context: str) -> List[BaseMessage]:
        """
        Create messages for the LLM.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
            
        Returns:
            List of messages
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Dựa trên thông tin sau:

{context}

Hãy trả lời câu hỏi: {question}""")
        ]
        
        return messages
    
    async def generate(self, question: str, context: str) -> str:
        """
        Generate a response asynchronously.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Generated response
        """
        logger.info(f"🤖 Generating response for: '{question[:50]}...'")
        logger.debug(f"Context length: {len(context)} chars")
        
        messages = self._create_messages(question, context)
        
        logger.debug(f"Sending {len(messages)} messages to LLM")
        response = await self.llm.ainvoke(messages)
        
        logger.info("✅ Response generated")
        logger.debug(f"Response length: {len(response.content)} chars")
        
        return response.content
    
    def generate_sync(self, question: str, context: str) -> str:
        """
        Generate a response synchronously.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Generated response
        """
        logger.info(f"🤖 Generating response for: '{question[:50]}...'")
        logger.debug(f"Context length: {len(context)} chars")
        
        messages = self._create_messages(question, context)
        
        logger.debug(f"Sending {len(messages)} messages to LLM")
        response = self.llm.invoke(messages)
        
        logger.info("✅ Response generated")
        logger.debug(f"Response length: {len(response.content)} chars")
        
        return response.content


if __name__ == "__main__":
    import asyncio
    
    # Test generator
    generator = GeneratorService()
    
    question = "Python là gì?"
    context = """Python là một ngôn ngữ lập trình phổ biến được sử dụng rộng rãi.
    Python có cú pháp đơn giản, dễ học và dễ đọc.
    Python được sử dụng trong nhiều lĩnh vực như web development, data science, machine learning."""
    
    # Test sync
    print("Testing sync generation...")
    response = generator.generate_sync(question, context)
    print(f"Response: {response}")
    
    # Test async
    async def test_async():
        print("\nTesting async generation...")
        response = await generator.generate(question, context)
        print(f"Response: {response}")
    
    asyncio.run(test_async())
