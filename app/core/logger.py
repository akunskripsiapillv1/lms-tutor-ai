# File: app/core/logger.py
"""
Konfigurasi logging terpusat untuk aplikasi.
Menggunakan format yang sama dengan ai-services untuk konsistensi.
"""
import logging
import sys
from pathlib import Path

# Import settings with lazy evaluation to avoid circular imports
def get_settings():
    from app.config.settings import settings
    return settings

# --- Konfigurasi ---
def get_log_config():
    settings = get_settings()
    log_dir = settings.log_dir
    log_level = "DEBUG" if settings.log_debug else "INFO"
    # Pastikan log directory ada
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, log_level

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

# Get configuration on import
LOG_DIR, LOG_LEVEL = get_log_config()

# --- Handler ---
def get_console_handler():
    """Stream handler untuk output ke konsol."""
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    return console_handler

def get_file_handler(log_file_name: str):
    """File handler untuk menulis ke file."""
    file_handler = logging.FileHandler(LOG_DIR / log_file_name, encoding="utf-8")
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    return file_handler

# --- Logger Factory ---
def setup_logger(name: str, log_file: str):
    """Membuat dan mengkonfigurasi logger."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False  # Mencegah duplikasi log

    if not logger.handlers:
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler(log_file))

    return logger

# --- Logger Instances ---
app_logger = setup_logger("app", "app.log")
api_logger = setup_logger("api", "api.log")
model_logger = setup_logger("model", "model.log")
cache_logger = setup_logger("cache", "cache.log")
telemetry_logger = setup_logger("telemetry", "telemetry.log")
chat_logger = setup_logger("chat", "chat.log")
rag_logger = setup_logger("rag", "rag.log")

app_logger.info(f"Logger dikonfigurasi. Level: {LOG_LEVEL}. Direktori: {LOG_DIR}")