"""
LLM (Large Language Model) initialization module.
Uses Qwen2.5-7B-Instruct via HuggingFace Inference API.
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
REPETITION_PENALTY = 1.03

logger.info(f"Initializing LLM with model: {MODEL_NAME}")
logger.debug(f"LLM config: max_new_tokens={MAX_NEW_TOKENS}, do_sample={DO_SAMPLE}, repetition_penalty={REPETITION_PENALTY}")

# Initialize HuggingFace LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id=MODEL_NAME,
    task="text-generation",
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=DO_SAMPLE,
    repetition_penalty=REPETITION_PENALTY,
)

# Create ChatHuggingFace wrapper for chat interface
logger.debug("Creating ChatHuggingFace wrapper")
chat_model = ChatHuggingFace(llm=llm)
logger.info("✅ LLM chat model initialized successfully")


if __name__ == "__main__":
    # Test the model
    response = chat_model.invoke("What is 2+2?")
    print(f"Response: {response.content}")
