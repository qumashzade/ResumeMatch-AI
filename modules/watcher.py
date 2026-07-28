"""
watcher.py — Qovluqları izləyir, yeni CV əlavə olunanda avtomatik təsnifatlandırır.

İzləyir:
  - Extra_CVs/ qovluğu
  - data/downloaded_cvs/ qovluğu (email-dən yüklənənlər)
  - İstənilən əlavə qovluq (--watch-folder ilə)

Yeni fayl aşkarlananda:
  1. Faylı oxuyur
  2. AI ilə sahəni müəyyən edir
  3. CVs/<sahə>/ qovluğuna kopyalayır
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Artıq emal edilmiş faylları yadda saxla
_processed_files = set()


def _classify_single_file(filepath: str, base_cv_folder: str, domains: list):
    """Bir CV faylını təsnifatlandırıb uyğun qovluğa kopyalayır."""
    from modules.data_loader import load_document
    from modules.ai_classifier import translate_if_needed, classify_cv_domain

    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return

    # Dublikat yoxla
    if filepath in _processed_files:
        return
    _processed_files.add(filepath)

    # Faylın tam yazılmasını gözlə (böyük fayllar üçün)
    time.sleep(1)

    # Yüklə
    doc = load_document(filepath)
    if not doc:
        logger.warning("Fayl oxuna bilmədi: %s", filename)
        return

    # Tərcümə et
    translated = translate_if_needed(doc.raw_text)

    # Sahəni müəyyən et
    domain = classify_cv_domain(translated, domains)

    # Qovluğa kopyala
    domain_folder = os.path.join(base_cv_folder, domain)
    os.makedirs(domain_folder, exist_ok=True)

    dest = os.path.join(domain_folder, filename)
    if not os.path.exists(dest):
        shutil.copy2(filepath, dest)
        print(f"  ✓ YENİ CV: {filename} → {domain}/")
        logger.info("Təsnifatlandırıldı: %s → %s", filename, domain)
    else:
        print(f"  ○ {filename} → {domain}/ (artıq var)")


def start_watching(watch_folders: list, base_cv_folder: str, domains: list):
    """
    Qovluqları izləməyə başlayır. Ctrl+C ilə dayandırılır.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("  watchdog quraşdırılmayıb! Quraşdırmaq üçün:")
        print("  pip install watchdog")
        return

    class CVHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            filepath = event.src_path
            ext = os.path.splitext(filepath)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                print(f"\n  Yeni fayl aşkarlandı: {os.path.basename(filepath)}")
                try:
                    _classify_single_file(filepath, base_cv_folder, domains)
                except Exception as e:
                    logger.error("Xəta: %s — %s", os.path.basename(filepath), e)

        def on_moved(self, event):
            if event.is_directory:
                return
            filepath = event.dest_path
            ext = os.path.splitext(filepath)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                print(f"\n  Fayl köçürüldü: {os.path.basename(filepath)}")
                try:
                    _classify_single_file(filepath, base_cv_folder, domains)
                except Exception as e:
                    logger.error("Xəta: %s — %s", os.path.basename(filepath), e)

    # Qovluqları yarat (yoxdursa)
    for folder in watch_folders:
        os.makedirs(folder, exist_ok=True)

    observer = Observer()
    handler = CVHandler()

    for folder in watch_folders:
        observer.schedule(handler, folder, recursive=False)
        print(f"  İzlənir: {os.path.abspath(folder)}/")

    observer.start()

    print(f"\n  Watcher aktiv — yeni CV-lər avtomatik təsnifatlandırılacaq.")
    print(f"  Dayandırmaq üçün: Ctrl+C")
    print(f"  {'─' * 50}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n  Watcher dayandırıldı.")
        observer.stop()

    observer.join()
