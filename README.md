# CV Matcher — Intelligent Resume Ranking Engine

AI-powered HR pipeline that automatically processes, classifies, and ranks CVs against job descriptions.

## Features

- **Email Integration** — IMAP ilə email-dən CV-ləri avtomatik yükləyir
- **AI CV Verification** — OpenAI ilə faylın CV olub-olmadığını yoxlayır
- **Multi-language Support** — İstənilən dildə CV/vakansiya (avtomatik tərcümə)
- **Domain Classification** — CV-ləri sahələrə bölüşdürür (Data Science, Backend, Frontend...)
- **OCR Support** — JPG/PNG şəkil CV-ləri Tesseract ilə oxuyur
- **Hybrid Ranking** — TF-IDF + BM25 + Skill Matching + Metadata
- **Duplicate Detection** — Eyni namizədin təkrar CV-lərini aşkarlayır
- **AI Analysis Report** — Hər namizəd üçün ətraflı hesabat
- **Interactive Dashboard** — Nəticələri brauzerdə görün
- **Caching** — Artıq yoxlanmış CV-ləri yenidən yoxlamır

## Setup

1. **Clone and install:**
```bash
git clone https://github.com/YOUR_USERNAME/cv-matcher.git
cd cv-matcher
pip install -r requirements.txt
```

2. **Install Tesseract OCR** (for image CVs):
   - Windows: https://github.com/tesseract-ocr/tesseract/releases
   - Mac: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

3. **Configure:**
```bash
cp config_example.py config.py
# Edit config.py with your API key and email credentials
```

## Usage

```bash
# Classify CVs into domain folders:
python main.py classify --source data

# Find top candidates for a vacancy:
python main.py match --top-k 5

# Full pipeline (email + classify + match + report + dashboard):
python main.py full-pipeline --top-k 5

# Without email:
python main.py full-pipeline --skip-email --top-k 5
```

## Project Structure

```
cv_matcher/
├── config_example.py       ← Copy to config.py and fill in
├── main.py                 ← Main entry point
├── dashboard.html           ← UI template
├── modules/
│   ├── email_reader.py     ← IMAP email reader
│   ├── data_loader.py      ← File loader (PDF/DOCX/TXT/IMG)
│   ├── ai_classifier.py    ← AI translation + classification
│   ├── preprocessing.py    ← Text processing + metadata
│   ├── embedding.py        ← TF-IDF / semantic vectors
│   ├── similarity.py       ← Hybrid ranking engine
│   └── duplicate_detector.py ← Duplicate CV detection
├── data/                   ← Job description file
├── CVs/                    ← Auto-created domain folders
├── Extra_CVs/              ← Manual CV additions (always checked)
└── Selected_Candidates/    ← Top CVs + report + dashboard
```

## Supported Formats

| Format | Source |
|--------|--------|
| .pdf   | Email, manual |
| .docx  | Email, manual |
| .txt   | Email, manual |
| .jpg/.png | Email, manual (OCR) |

## Ranking Signals

| Signal | Weight | Description |
|--------|--------|-------------|
| Embedding | 45% | TF-IDF semantic similarity |
| BM25 | 20% | Keyword frequency overlap |
| Skills | 25% | Direct skill match ratio |
| Metadata | 10% | Education + experience bonus |

## License

MIT
