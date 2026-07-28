"""
email_reader.py — HR email hesabından CV-ləri avtomatik yükləyir.

IMAP protokolu ilə emailləri oxuyur, əlavə edilmiş faylları (PDF, DOCX, TXT)
yükləyir və lokal qovluğa saxlayır.
"""

import os
import email
import imaplib
import logging
from email.header import decode_header
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".doc", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _extract_text_for_check(filepath: str, ext: str) -> str:
    """Fayldan mətn çıxarır (CV yoxlaması üçün)."""
    try:
        if ext == ".pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext in (".docx", ".doc"):
            from docx import Document as DocxDocument
            doc = DocxDocument(filepath)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext in (".txt", ".text", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"):
            try:
                import pytesseract
                from PIL import Image, ImageEnhance
                import shutil
                if not shutil.which("tesseract"):
                    common = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
                    ]
                    for p in common:
                        if os.path.isfile(p):
                            pytesseract.pytesseract.tesseract_cmd = p
                            break
                img = Image.open(filepath).convert("L")
                img = ImageEnhance.Contrast(img).enhance(2.0)
                return pytesseract.image_to_string(img, lang="eng")
            except Exception:
                return ""
    except Exception:
        return ""
    return ""


def _is_cv_document(text: str, filename: str, email_subject: str) -> bool:
    """
    OpenAI ilə faylın CV/resume olub-olmadığını yoxlayır.
    Faktura, qəbz, reklam, hesabat və s. rədd edilir.
    """
    import json
    import urllib.request
    import os
    import sys

    # config.py-dan API key oxu
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from config import OPENAI_API_KEY
        api_key = OPENAI_API_KEY
    except ImportError:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key or api_key == "your_openai_api_key_here":
        # API key yoxdursa, sadə keyword yoxlaması et
        cv_hints = {"experience", "education", "skills", "work", "university",
                    "cv", "resume", "developer", "engineer", "projects",
                    "bachelor", "master", "proficiency", "certification"}
        non_cv_hints = {"invoice", "receipt", "order", "payment", "shipping",
                        "subscription", "unsubscribe", "promo", "discount", "qəbz"}
        text_lower = text.lower()
        cv_count = sum(1 for h in cv_hints if h in text_lower)
        non_cv_count = sum(1 for h in non_cv_hints if h in text_lower)
        return cv_count >= 3 and cv_count > non_cv_count

    # AI ilə yoxla
    sample = text[:1500]
    prompt = (
        f"Filename: {filename}\n"
        f"Email subject: {email_subject}\n\n"
        f"Document content (first 1500 chars):\n{sample}\n\n"
        "Is this document a CV / resume / curriculum vitae of a job candidate? "
        "It is NOT a CV if it's an invoice, receipt, newsletter, report, advertisement, "
        "ticket, contract, or any other non-CV document.\n"
        "Reply ONLY 'yes' or 'no'."
    )

    try:
        payload = json.dumps({
            "model": "gpt-4o-mini",
            "max_tokens": 10,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            answer = result["choices"][0]["message"]["content"].strip().lower()
            return answer.startswith("yes")

    except Exception as e:
        logger.warning("AI CV yoxlaması uğursuz: %s — keyword ilə yoxlanılır", e)
        # Fallback: keyword yoxlaması
        cv_hints = {"experience", "education", "skills", "work", "university"}
        return sum(1 for h in cv_hints if h in text.lower()) >= 3


@dataclass
class EmailRecord:
    """Bir emailin məlumatları."""
    sender: str
    subject: str
    received_date: str
    attachments: List[str] = field(default_factory=list)  # yüklənmiş fayl yolları


def _decode_str(raw) -> str:
    """Email header-lərini decode edir."""
    if raw is None:
        return ""
    decoded_parts = decode_header(raw)
    parts = []
    for data, charset in decoded_parts:
        if isinstance(data, bytes):
            parts.append(data.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(data)
    return " ".join(parts)


def _sanitize_filename(name: str) -> str:
    """Fayl adından təhlükəli simvolları təmizləyir."""
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name.strip()


def fetch_cvs_from_email(
    imap_server: str,
    imap_port: int,
    email_address: str,
    password: str,
    mailbox: str = "INBOX",
    search_criteria: str = "UNSEEN",
    download_folder: str = "downloaded_cvs",
    max_emails: int = 0,
) -> List[EmailRecord]:
    """
    IMAP ilə email hesabına qoşulur, CV əlavələrini yükləyir.

    Args:
        max_emails: Maksimum neçə email yoxlansın (0 = limit yoxdur)

    Returns:
        List[EmailRecord] — yüklənmiş emaillərin siyahısı
    """
    os.makedirs(download_folder, exist_ok=True)

    # Artıq yüklənmiş faylları keşlə
    already_downloaded = set()
    if os.path.isdir(download_folder):
        already_downloaded = {f for f in os.listdir(download_folder) if os.path.isfile(os.path.join(download_folder, f))}
    # Sahə qovluqlarındakı CV-ləri də yoxla
    cvs_base = os.path.join(os.path.dirname(download_folder), "..", "CVs")
    if os.path.isdir(cvs_base):
        for domain in os.listdir(cvs_base):
            domain_path = os.path.join(cvs_base, domain)
            if os.path.isdir(domain_path):
                for f in os.listdir(domain_path):
                    already_downloaded.add(f)
    if already_downloaded:
        logger.info("%d fayl artıq yüklənib (atlanacaq)", len(already_downloaded))

    records: List[EmailRecord] = []

    try:
        # ── IMAP qoşulma ──────────────────────────────────────────────
        logger.info("Email serverə qoşulur: %s:%d", imap_server, imap_port)
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(email_address, password)
        mail.select(mailbox)
        logger.info("Email hesabına uğurla qoşuldu: %s", email_address)

        # ── Emailləri axtar ───────────────────────────────────────────
        status, message_ids = mail.search(None, search_criteria)
        if status != "OK" or not message_ids[0]:
            logger.info("Heç bir email tapılmadı (criteria: %s)", search_criteria)
            mail.logout()
            return records

        ids = message_ids[0].split()
        # Son N emaili yoxla (ən yenilər sonda olur)
        if max_emails > 0 and len(ids) > max_emails:
            logger.info("%d email tapıldı, son %d-i yoxlanılır", len(ids), max_emails)
            ids = ids[-max_emails:]
        else:
            logger.info("%d email tapıldı", len(ids))

        # ── Hər emaili işlə ──────────────────────────────────────────
        for msg_id in ids:
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])

            # Metadata çıxar
            sender = _decode_str(msg.get("From", ""))
            subject = _decode_str(msg.get("Subject", ""))
            date_str = _decode_str(msg.get("Date", ""))

            record = EmailRecord(
                sender=sender,
                subject=subject,
                received_date=date_str,
            )

            logger.info("Email: '%s' — %s", subject, sender)

            # ── Əlavələri yüklə ──────────────────────────────────────
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition", ""))
                if "attachment" not in content_disposition:
                    continue

                filename = part.get_filename()
                if filename:
                    filename = _decode_str(filename)
                    filename = _sanitize_filename(filename)

                    # Dəstəklənən formatmı?
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        logger.info("  Atlandı (format dəstəklənmir): %s", filename)
                        continue

                    # Artıq yüklənibmi?
                    if filename in already_downloaded:
                        logger.info("  Atlandı (artıq var): %s", filename)
                        continue

                    # Faylın məzmununu oxu və CV olub-olmadığını AI ilə yoxla
                    file_data = part.get_payload(decode=True)
                    if not file_data:
                        continue

                    # Müvəqqəti fayla yaz ki, mətn çıxara bilək
                    import tempfile
                    temp_path = os.path.join(tempfile.gettempdir(), filename)
                    with open(temp_path, "wb") as tmp:
                        tmp.write(file_data)

                    # Mətni çıxar
                    extracted_text = _extract_text_for_check(temp_path, ext)

                    # Təmizlə
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

                    if not extracted_text.strip():
                        logger.info("  Atlandı (mətn çıxarıla bilmədi): %s", filename)
                        continue

                    # AI ilə CV olub-olmadığını yoxla
                    if not _is_cv_document(extracted_text, filename, subject):
                        logger.info("  Atlandı (CV deyil): %s", filename)
                        continue

                    # Eyni adlı fayl varsa, tarix əlavə et
                    filepath = os.path.join(download_folder, filename)
                    if os.path.exists(filepath):
                        name, ext = os.path.splitext(filename)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{name}_{timestamp}{ext}"
                        filepath = os.path.join(download_folder, filename)

                    # Faylı yaz
                    with open(filepath, "wb") as f:
                        f.write(file_data)

                    record.attachments.append(filepath)
                    logger.info("  ✓ CV yükləndi: %s", filename)

            if record.attachments:
                records.append(record)

        mail.logout()
        logger.info(
            "Cəmi: %d emaildən %d CV yükləndi",
            len(ids),
            sum(len(r.attachments) for r in records),
        )

    except imaplib.IMAP4.error as e:
        logger.error("IMAP xətası: %s", e)
        logger.error(
            "Gmail istifadə edirsinizsə: 2FA aktiv edin və App Password yaradın.\n"
            "  https://myaccount.google.com/apppasswords"
        )
    except Exception as e:
        logger.error("Email oxuma xətası: %s", e)

    return records
