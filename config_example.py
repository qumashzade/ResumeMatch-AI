"""
config.py — Bütün sistem konfiqurasiyaları.

İSTİFADƏ:
  1. Bu faylı "config.py" adıyla kopyalayın
  2. Aşağıdakı dəyərləri öz məlumatlarınızla doldurun
  3. config.py heç vaxt GitHub-a yüklənməyəcək (.gitignore-da var)
"""

# ─── OpenAI API Key ────────────────────────────────────────────────────────
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"

# ─── Email Konfiqurasiyası (IMAP — email oxumaq üçün) ─────────────────────
EMAIL_CONFIG = {
    "imap_server": "imap.gmail.com",       # Gmail üçün. Outlook: imap-mail.outlook.com
    "imap_port": 993,
    "email": "your_email@gmail.com",        # HR email ünvanı
    "password": "your_app_password",        # Gmail App Password (2FA aktiv olmalıdır)
    "mailbox": "INBOX",
    "search_criteria": "ALL",               # ALL, UNSEEN, 'SINCE "01-Jul-2026"'
    "max_emails": 100,                      # Son neçə emaili yoxlasın (0 = limit yoxdur)
}

# ─── SMTP Konfiqurasiyası (namizədlərə email göndərmə) ────────────────────
SMTP_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": "your_email@gmail.com",
    "password": "your_app_password",
    "sender_name": "HR Department",
}

# ─── Müsahibəyə dəvət email şablonu ──────────────────────────────────────
INVITE_EMAIL_TEMPLATE = {
    "subject": "Interview Invitation — {company_name}",
    "body": """Dear {candidate_name},

Thank you for your application. We have reviewed your CV and would like to invite you for an interview for the {job_title} position.

Interview Details:
- Date: {interview_date}
- Format: {interview_format}

Please confirm your availability by replying to this email.

Best regards,
{sender_name}
{company_name}
""",
    "company_name": "Your Company",
    "interview_date": "TBD",
    "interview_format": "Online (Google Meet)",
}

# ─── Avtomatik email yoxlama (watch rejimi) ───────────────────────────────
WATCH_CONFIG = {
    "interval_minutes": 30,
    "auto_classify": True,
    "auto_match": False,
}

# ─── Qovluq yolları ───────────────────────────────────────────────────────
BASE_CV_FOLDER = "CVs"
EXTRA_CV_FOLDER = "Extra_CVs"
SELECTED_FOLDER = "Selected_Candidates"
DATA_FOLDER = "data"
VACANCIES_FOLDER = "vacancies"
JOB_DESCRIPTION_FILE = "data/job_description.txt"

# ─── Sistem parametrləri ──────────────────────────────────────────────────
TOP_K = 5
EMBEDDING_MODE = "tfidf"

# ─── Tanınan sahələr ──────────────────────────────────────────────────────
DOMAINS = [
    "Data_Science",
    "Backend_Development",
    "Frontend_Development",
    "AI_ML_Engineering",
    "DevOps",
    "UI_UX_Design",
    "Cybersecurity",
    "Mobile_Development",
    "QA_Engineering",
    "Product_Management",
    "Business_Analysis",
    "Data_Engineering",
    "Cloud_Engineering",
    "Full_Stack_Development",
    "Other",
]
