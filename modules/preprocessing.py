"""
preprocessing.py — CV və iş elanı mətnlərini təmizləyir, strukturlaşdırır.

Funksiyalar:
  1. Mətn normalizasiyası
  2. Bölmə aşkarlaması (skills, education, experience, projects)
  3. Metadata çıxarma (bacarıqlar, təhsil səviyyəsi, təcrübə illəri)
  4. Bölmə ağırlıqlı mətn hazırlama (skills > hobbies)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Bölmə başlıq nümunələri ──────────────────────────────────────────────
SECTION_PATTERNS = {
    "skills": r"(?:technical\s+)?skills?|proficiencies|technologies|tech\s+stack|competencies|technical\s+proficiencies",
    "education": r"education|academic|qualifications|degrees?",
    "experience": r"(?:work\s+)?experience|employment|professional\s+background|career",
    "projects": r"projects|portfolio|personal\s+projects",
    "certifications": r"certifications?|certificates?|credentials",
    "publications": r"publications?|papers?|research\s+output",
    "awards": r"awards?|achievements?|honors?|honours?",
    "interests": r"interests?|hobbies|extracurricular|activities",
}

# ─── Texniki bacarıqlar siyahısı ──────────────────────────────────────────
KNOWN_SKILLS = {
    "python", "r", "sql", "java", "scala", "julia", "c++", "c#", "javascript",
    "typescript", "matlab", "perl", "bash", "vba", "go", "rust", "ruby", "php",
    "swift", "kotlin", "dart", "objective-c",
    "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", "xgboost",
    "lightgbm", "catboost", "h2o", "hugging face", "transformers", "optuna",
    "pandas", "numpy", "scipy", "polars", "dask",
    "spark", "pyspark", "hadoop", "kafka", "flink", "hive", "airflow",
    "matplotlib", "seaborn", "plotly", "tableau", "power bi", "grafana",
    "streamlit", "dash", "looker",
    "aws", "gcp", "azure", "sagemaker", "bigquery", "redshift",
    "docker", "kubernetes", "mlflow", "kubeflow", "dvc", "git", "ci/cd",
    "terraform", "ansible", "jenkins",
    "postgresql", "mysql", "mongodb", "redis", "cassandra", "elasticsearch",
    "sqlite", "dynamodb",
    "nltk", "spacy", "gensim",
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "fastapi", "spring", "rails",
    "deep learning", "neural networks", "nlp", "computer vision",
    "reinforcement learning", "a/b testing", "statistical analysis",
    "hypothesis testing", "bayesian", "time series",
    "figma", "sketch", "adobe xd", "selenium", "cypress", "jest",
    "jira", "confluence", "agile", "scrum",
}

EDUCATION_LEVELS = {
    "phd": 4, "doctorate": 4, "dphil": 4,
    "msc": 3, "master": 3, "mba": 3, "msci": 3, "ma ": 3, "meng": 3,
    "bsc": 2, "bachelor": 2, "ba ": 2, "beng": 2,
    "hdip": 1, "diploma": 1, "certificate": 1,
}

SECTION_WEIGHTS = {
    "skills": 3.0,
    "experience": 2.5,
    "education": 2.0,
    "projects": 2.0,
    "certifications": 1.5,
    "publications": 1.5,
    "awards": 1.0,
    "interests": 0.3,
}


@dataclass
class ProcessedDocument:
    """Emal edilmiş sənəd."""
    filename: str
    clean_text: str
    weighted_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    extracted_skills: List[str] = field(default_factory=list)
    education_level: Optional[str] = None
    education_score: int = 0
    years_experience: float = 0.0
    skill_tokens: List[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_sections(text: str) -> Dict[str, str]:
    lines = text.split("\n")
    sections: Dict[str, str] = {}
    current_section = "header"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip().lower()
        matched = False
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(rf"^(?:{pattern})\s*[:\-]?\s*$", stripped, re.IGNORECASE):
                if current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = section_name
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()
    return sections


def _extract_skills(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        if len(skill) <= 2:
            if re.search(rf"\b{re.escape(skill)}\b", text_lower):
                found.append(skill)
        else:
            if skill in text_lower:
                found.append(skill)
    return sorted(set(found))


def _detect_education_level(text: str) -> Tuple[Optional[str], int]:
    text_lower = text.lower()
    best_level = None
    best_score = 0
    for keyword, score in EDUCATION_LEVELS.items():
        if keyword in text_lower and score > best_score:
            best_level = keyword.strip()
            best_score = score
    return best_level, best_score


def _estimate_experience_years(text: str) -> float:
    MONTHS = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4,
        "june": 6, "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12, "summer": 6,
    }
    pattern = re.compile(
        r"(?:(\w+)\s+)?(\d{4})\s*[-–—]\s*(?:(\w+)\s+)?(\d{4}|present|current|now)",
        re.IGNORECASE,
    )
    total = 0.0
    for m in pattern.finditer(text):
        s_month_str, s_year, e_month_str, e_year_str = m.groups()
        s_year = int(s_year)
        s_month = MONTHS.get((s_month_str or "").lower(), 1)
        if e_year_str.isdigit():
            e_year = int(e_year_str)
            e_month = MONTHS.get((e_month_str or "").lower(), 12)
        else:
            e_year, e_month = 2026, 5
        years = (e_year - s_year) + (e_month - s_month) / 12.0
        total += max(0, years)
    return round(total, 1)


def _build_weighted_text(sections: Dict[str, str]) -> str:
    parts = []
    for section_name, content in sections.items():
        weight = SECTION_WEIGHTS.get(section_name, 1.0)
        repeat_count = max(1, int(weight))
        for _ in range(repeat_count):
            parts.append(content)
    return " ".join(parts)


def _tokenize_for_bm25(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\#\s\-]", " ", text)
    tokens = text.split()
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "this", "that", "these", "those", "i", "we", "you", "he", "she",
        "it", "they", "me", "him", "her", "us", "them", "my", "our",
    }
    return [t for t in tokens if t not in stopwords and (len(t) > 1 or t in ("r", "c"))]


def preprocess_document(filename: str, raw_text: str, translate_fn=None) -> ProcessedDocument:
    """
    Tam emal pipeline-ı.
    translate_fn verilsə, əvvəlcə tərcümə edir.
    """
    text = translate_fn(raw_text) if translate_fn else raw_text

    clean = normalize_text(text)
    sections = _detect_sections(text)
    skills = _extract_skills(text)
    edu_level, edu_score = _detect_education_level(text)
    years = _estimate_experience_years(text)
    weighted = _build_weighted_text(sections)
    tokens = _tokenize_for_bm25(clean)

    return ProcessedDocument(
        filename=filename,
        clean_text=clean,
        weighted_text=weighted,
        sections=sections,
        extracted_skills=skills,
        education_level=edu_level,
        education_score=edu_score,
        years_experience=years,
        skill_tokens=tokens,
    )


def preprocess_all(documents: list, translate_fn=None) -> List[ProcessedDocument]:
    processed = []
    for doc in documents:
        p = preprocess_document(doc.filename, doc.raw_text, translate_fn)
        processed.append(p)
        logger.info(
            "Emal: %s — %d bacarıq, təhsil=%s, ~%.0f il təcrübə",
            doc.filename, len(p.extracted_skills),
            p.education_level, p.years_experience,
        )
    return processed
