"""
PDF Processing Service for Vietnamese Law Documents.
Processes multiple PDF files from a folder and outputs to structured JSON files.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
import pymupdf as fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 

class PDFProcessingService:
    """Service for processing PDF files to structured chunks."""
    
    def __init__(self, extraction_method: str = "fitz", max_chunk_length: int = 800):
        """
        Initialize PDF Processing Service.
        
        Args:
            extraction_method: Method to extract text ("pypdf2" or "fitz")
            max_chunk_length: Maximum length for each chunk
        """
        self.extraction_method = extraction_method
        self.max_chunk_length = max_chunk_length
        
        # Regex patterns for identifying chapters and articles
        self.chapter_pattern = r"(Chương\s+[IVXLCDM]+\s*[\n\r]*[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰ\s]+?)(?=\s*Điều|$)"
        self.article_pattern = r"(Điều\s+\d+[a-z]?\.\s*[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰ][^\n]+)"
    
    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyPDF2."""
        text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text()
        return text

    def extract_text_with_fitz(self, pdf_path: str) -> str:
        """Extract text from a PDF file using fitz (PyMuPDF)."""
        text = ""
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        return text
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using configured method.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if self.extraction_method == "pypdf2":
            return self.extract_text_with_pypdf2(pdf_path)
        elif self.extraction_method == "fitz":
            return self.extract_text_with_fitz(pdf_path)
        else:
            raise ValueError(f"Unsupported extraction method: {self.extraction_method}")

    def preprocess_and_chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Preprocess the text and split it into chunks with titles and contexts.
        - Titles include both the current chapter and article.
        - Contexts contain the text under each article.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            List of chunks with title and context
        """
        # Split the text into sections based on chapters and articles
        sections = re.split(
            f"({self.chapter_pattern}|{self.article_pattern})", 
            text, 
            flags=re.DOTALL
        )
        
        # Initialize variables for processing
        current_chapter = None
        current_article = None
        chunks = []
        buffer = ""
        
        for section in sections:
            # Skip None or empty sections
            if section is None or not section.strip():
                continue
            
            # Check if the section is a chapter title
            chapter_match = re.match(self.chapter_pattern, section)
            if chapter_match:
                # If there's a previous article, save its content as a chunk
                if current_article and buffer.strip():
                    chunk = {
                        "title": f"{current_article} {current_chapter}",
                        "context": buffer.strip()
                    }
                    chunks.append(chunk)
                
                # Update the current chapter
                current_chapter = section.strip()
                current_article = None  # Reset article when a new chapter starts
                buffer = ""  # Reset buffer for new chapter
                continue
            
            # Check if the section is an article title
            article_match = re.match(self.article_pattern, section)
            if article_match:
                # If there's a previous article, save its content as a chunk
                if current_article and buffer.strip():
                    chunk = {
                        "title": f"{current_article} {current_chapter}",
                        "context": buffer.strip()
                    }
                    chunks.append(chunk)
                
                # Update the current article
                current_article = section.strip()
                buffer = ""  # Reset buffer for new article
                continue
            
            # If it's neither a chapter nor an article, it's part of the current article's content
            if current_article:
                buffer += " " + section.strip()
        
        # Add the last chunk if there's any remaining content
        if current_article and buffer.strip():
            chunk = {
                "title": f"{current_article} {current_chapter}",
                "context": buffer.strip()
            }
            chunks.append(chunk)
        
        return chunks

    def split_long_context(self, title: str, context: str) -> List[Dict[str, str]]:
        """
        Split long context into smaller chunks using RecursiveCharacterTextSplitter.
        Each smaller chunk retains the same title.
        
        Args:
            title: Title for the chunks
            context: Long context to split
            
        Returns:
            List of smaller chunks with same title
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_length,
            chunk_overlap=300,  # Overlap to ensure continuity between chunks
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        sub_chunks = splitter.split_text(context)
        return [{"title": title, "context": sub_chunk.strip()} for sub_chunk in sub_chunks]

    def process_single_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed chunks
        """
        print(f"\n📄 Processing: {pdf_path}")
        
        # Step 1: Extract text from the PDF
        raw_text = self.extract_text(pdf_path)
        print(f"  ✓ Extracted {len(raw_text)} characters")
        
        # Step 2: Preprocess and chunk the text
        chunks = self.preprocess_and_chunk_text(raw_text)
        print(f"  ✓ Created {len(chunks)} initial chunks")
        
        # Step 3: Split long contexts into smaller chunks
        final_chunks = []
        for chunk in chunks:
            title = chunk["title"]
            context = chunk["context"]
            if len(context) > self.max_chunk_length:
                sub_chunks = self.split_long_context(title, context)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        print(f"  ✓ Final chunks: {len(final_chunks)}")
        return final_chunks

    def process_folder(
        self, 
        input_folder: str, 
        output_folder: str,
        combine_output: bool = True
    ) -> Dict[str, Any]:
        """
        Process all PDF files in a folder.
        
        Args:
            input_folder: Path to folder containing PDF files
            output_folder: Path to output folder for JSON files
            combine_output: If True, combine all chunks into single file
            
        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output folder if not exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {input_folder}")
            return {"processed": 0, "total_chunks": 0}
        
        print(f"\n{'='*60}")
        print(f"🚀 PDF Processing Service")
        print(f"{'='*60}")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Found {len(pdf_files)} PDF file(s)")
        
        all_chunks = []
        processed_files = []
        
        # Process each PDF file
        for pdf_file in pdf_files:
            try:
                chunks = self.process_single_pdf(str(pdf_file))
                
                if not combine_output:
                    # Save each PDF's output separately
                    output_file = output_path / f"{pdf_file.stem}.json"
                    self.save_to_json(chunks, str(output_file))
                    print(f"  ✓ Saved to: {output_file.name}")
                
                all_chunks.extend(chunks)
                processed_files.append(pdf_file.name)
                
            except Exception as e:
                print(f"  ❌ Error processing {pdf_file.name}: {e}")
        
        # Save combined output if requested
        if combine_output and all_chunks:
            combined_file = output_path / "combined_output.json"
            self.save_to_json(all_chunks, str(combined_file))
            print(f"\n✅ Combined output saved to: {combined_file}")
        
        # Statistics
        stats = {
            "processed_files": len(processed_files),
            "total_pdfs": len(pdf_files),
            "total_chunks": len(all_chunks),
            "files": processed_files,
            "output_folder": str(output_path)
        }
        
        print(f"\n{'='*60}")
        print(f"📊 Processing Summary:")
        print(f"  Processed: {stats['processed_files']}/{stats['total_pdfs']} files")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"{'='*60}\n")
        
        return stats

    def save_to_json(self, data: List[Dict[str, str]], output_file: str):
        """
        Save processed data to JSON file.
        
        Args:
            data: List of chunks to save
            output_file: Path to output JSON file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)




# Legacy functions for backward compatibility (deprecated)
def extract_text_with_pypdf2(pdf_path):
    """Extract text from a PDF file using PyPDF2. (Deprecated - use PDFProcessingService)"""
    service = PDFProcessingService(extraction_method="pypdf2")
    return service.extract_text(pdf_path)


def extract_text_with_fitz(pdf_path):
    """Extract text from a PDF file using fitz (PyMuPDF). (Deprecated - use PDFProcessingService)"""
    service = PDFProcessingService(extraction_method="fitz")
    return service.extract_text(pdf_path)


def preprocess_and_chunk_text(text):
    """Preprocess and chunk text. (Deprecated - use PDFProcessingService)"""
    service = PDFProcessingService()
    return service.preprocess_and_chunk_text(text)


def split_long_context(title, context, max_length=800):
    """Split long context. (Deprecated - use PDFProcessingService)"""
    service = PDFProcessingService(max_chunk_length=max_length)
    return service.split_long_context(title, context)


def process_pdf(pdf_path, output_json, extraction_method="fitz"):
    """Process PDF to JSON. (Deprecated - use PDFProcessingService)"""
    service = PDFProcessingService(extraction_method=extraction_method)
    chunks = service.process_single_pdf(pdf_path)
    service.save_to_json(chunks, output_json)


def save_to_json(data, output_file):
    """Save to JSON. (Deprecated - use PDFProcessingService)"""
    service = PDFProcessingService()
    service.save_to_json(data, output_file)


# Main execution
if __name__ == "__main__":
    """
    Main execution: Process all PDFs from 'data' folder to 'output_data' folder.
    """
    # Get current directory
    current_dir = Path(__file__).parent.parent
    
    # Define paths
    input_folder = current_dir / "data"
    output_folder = current_dir / "output_data"
    
    # Initialize service
    service = PDFProcessingService(
        extraction_method="fitz",  # or "pypdf2"
        max_chunk_length=800
    )
    
    # Process all PDFs in folder
    stats = service.process_folder(
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        combine_output=True  # Creates combined_output.json
    )
    
    print(f"\n✅ Processing completed!")
    print(f"📁 Output saved to: {stats['output_folder']}")
