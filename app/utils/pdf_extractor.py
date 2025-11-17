# File: app/utils/pdf_extractor.py
"""
Utility untuk mengekstrak dan memecah teks dari file PDF.
(Dipindah dari utils.py)
"""
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Impor dari struktur baru
from app.core.config import settings
from app.core.exceptions import PDFExtractionError
from app.core.logger import app_logger

class PDFExtractor:
    """Utility for extracting and splitting text from PDF files."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract(self, file_path: Path) -> List[Document]:
        """
        Extract and split text from PDF.
        """
        try:
            app_logger.info(f"Mengekstrak teks dari {file_path.name}")
            
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            if not pages:
                raise PDFExtractionError(
                    "Tidak ada halaman yang diekstrak dari PDF",
                    details={"file": str(file_path)}
                )
            
            chunks = self.text_splitter.split_documents(pages)
            
            if not chunks:
                raise PDFExtractionError(
                    "Tidak ada chunk teks yang dihasilkan dari PDF",
                    details={"file": str(file_path), "pages": len(pages)}
                )
            
            app_logger.info(
                f"Ekstrak {len(pages)} halaman, menghasilkan {len(chunks)} chunk"
            )
            
            return chunks
            
        except PDFExtractionError:
            raise
        except Exception as e:
            app_logger.error(f"Error mengekstrak PDF {file_path}: {e}")
            raise PDFExtractionError(
                f"Gagal mengekstrak teks dari PDF: {e}",
                details={"file": str(file_path), "error": str(e)}
            )

# Global instance
pdf_extractor = PDFExtractor()