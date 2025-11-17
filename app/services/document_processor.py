# File: app/services/document_processor.py
"""
Service untuk pipeline pemrosesan dokumen (Chunking & Embedding).
Ini adalah pipeline data bersama untuk fitur-fitur
yang membutuhkan RAG (seperti AQAG).
"""
from pathlib import Path
from typing import List

# Impor dari struktur baru
from app.utils.pdf_extractor import pdf_extractor
from app.repositories.unified_vector_store import unified_vector_store
# from app.db.db_models import Document, Course
from app.core.logger import app_logger
from app.core.exceptions import FileProcessingError
from app.config.settings import settings 
from langchain_core.documents import Document as LangchainDocument

class DocumentProcessor:
    """
    Menangani alur: Cek -> Ekstrak -> Chunk -> Embed -> Store.
    """
    
    async def ensure_documents_are_processed(
        self,
        file_paths: List[Path],
        material_ids: List[str],
        course_id: str = "default",
        db_session=None,
        file_names: List[str] = None
    ) -> dict:
        """
        Memastikan semua dokumen ada di Vector Store.
        Melewatkan dokumen yang sudah ada
        """
        if not file_paths or len(file_paths) != len(material_ids):
            raise FileProcessingError("Jumlah file_paths dan material_ids harus cocok dan tidak kosong")

        stats = {"processed": 0, "skipped": 0}
        chunks_to_add = []
        processed_documents = []  # Track documents for status updates

        app_logger.info(f"Memulai pemrosesan dokumen untuk course_id: {course_id}")

        # Create zip with file_names if available
        files_to_process = []
        if file_names and len(file_names) == len(file_paths):
            files_to_process = zip(file_paths, material_ids, file_names)
        else:
            files_to_process = zip(file_paths, material_ids, [None] * len(file_paths))

        for file_path, material_id, original_filename in files_to_process:
            document_record = None  # Initialize for error handling scope

            try:
                # LANGKAH KUNCI: Cek apakah hash ini sudah ada
                exists = await unified_vector_store.check_document_exists(material_id)

                if exists:
                    app_logger.info(f"SKIP: Dokumen {file_path.name} ({material_id[:7]}...) sudah ada.")
                    stats["skipped"] += 1
                    continue

                # Jika tidak ada, proses
                app_logger.info(f"PROSES: Dokumen {file_path.name} ({material_id[:7]}...)")

                # 0. Create Document record in database
                if db_session:
                    try:
                        # Check if document already exists in database
                        existing_doc = db_session.query(Document).filter(
                            Document.md5_hash == material_id
                        ).first()

                        if not existing_doc:
                            # Get valid course_id if default doesn't exist
                            valid_course_id = course_id
                            if course_id == "default":
                                # Get first available course as default
                                first_course = db_session.query(Course).first()
                                if first_course:
                                    valid_course_id = first_course.course_id
                                    app_logger.info(f"Using course '{valid_course_id}' instead of 'default'")
                                else:
                                    # Create default course if none exists
                                    from app.db.db_models import User
                                    default_user = db_session.query(User).first()
                                    new_course = Course(
                                        title="Default Course",
                                        instructor_id=default_user.user_id if default_user else None
                                    )
                                    db_session.add(new_course)
                                    db_session.commit()
                                    db_session.refresh(new_course)
                                    valid_course_id = new_course.course_id
                                    app_logger.info(f"Created new default course: {valid_course_id}")

                            # Use original filename for title, file_path for path
                            title_name = original_filename or file_path.name
                            app_logger.info(f"Creating document with title: {title_name} from path: {file_path.name}")

                            document_record = Document.create_from_upload(
                                db=db_session,
                                course_id=valid_course_id,
                                file_name=title_name,
                                material_id=material_id,
                                file_size=file_path.stat().st_size,
                                file_type="pdf"
                            )
                            app_logger.info(f"Created document record: {document_record.md5_hash} with title: {title_name}")
                        else:
                            # Update existing document to processing status
                            document_record = existing_doc
                            document_record.update_status(db_session, "processing")
                            app_logger.info(f"Updated existing document to processing: {document_record.md5_hash}")

                        # Add to processed documents list for later status update
                        processed_documents.append(document_record)
                    except Exception as db_error:
                        app_logger.warning(f"Failed to create/update document record: {db_error}")

                # 1. Ekstrak & Chunk
                chunks = pdf_extractor.extract(file_path)
                
                # 2. Tambahkan metadata penting
                for chunk in chunks:
                    chunk.metadata["course_id"] = course_id
                    chunk.metadata["material_id"] = material_id
                    chunk.metadata["source_file"] = file_path.name
                
                chunks_to_add.extend(chunks)
                stats["processed"] += 1

            except Exception as e:
                app_logger.error(f"Gagal memproses file {file_path.name}: {e}")

                # Update document status to failed if record was created
                if db_session and document_record:
                    try:
                        document_record.update_status(db_session, "failed")
                        app_logger.info(f"Updated document to failed status: {document_record.md5_hash}")
                    except Exception as status_error:
                        app_logger.warning(f"Failed to update document to failed status: {status_error}")

                continue
        
        # 3. Simpan semua chunk baru ke Redis dalam satu batch
        if chunks_to_add:
            await unified_vector_store.add_documents(chunks_to_add)
            app_logger.info(f"Menyimpan {len(chunks_to_add)} chunks baru ke Vector Store.")

        # 4. Update document statuses to completed
        if db_session and processed_documents:
            try:
                for doc in processed_documents:
                    doc.update_status(
                        db=db_session,
                        status="completed",
                        has_embeddings=True,
                        embedding_model=settings.openai_embedding_model
                    )
                app_logger.info(f"Updated {len(processed_documents)} documents to completed status")
            except Exception as update_error:
                app_logger.warning(f"Failed to update document statuses: {update_error}")

        summary_msg = f"Proses selesai: {stats['processed']} file diproses, {stats['skipped']} file dilewati."
        app_logger.info(summary_msg)
        return {"message": summary_msg, **stats}

# Global instance
document_processor = DocumentProcessor()