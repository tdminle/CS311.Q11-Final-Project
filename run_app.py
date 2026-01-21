"""
Run Streamlit app for Vietnamese Law RAG System
"""
import subprocess
import sys
import os

def main():
    """Run the Streamlit app."""
    print("🚀 Starting Vietnamese Law RAG System...")
    print("📍 Opening Streamlit app at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Đảm bảo chạy từ thư mục dự án
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Sử dụng uv run để đảm bảo môi trường đúng
        subprocess.run([
            "uv", "run", "streamlit", "run",
            "ui/app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        sys.exit(0)
    except FileNotFoundError:
        # Fallback nếu uv không có trong PATH
        print("⚠️  'uv' not found, using python directly...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "ui/app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])

if __name__ == "__main__":
    main()
