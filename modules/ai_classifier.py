"""
ai_classifier.py — OpenAI API ilə CV-ləri sahələrə görə təsnifatlandırır və tərcümə edir.

Funksiyalar:
  1. CV-nin sahəsini müəyyən edir (Data Science, Backend, Frontend, və s.)
  2. Qeyri-ingilis mətnləri ingiliscəyə tərcümə edir
  3. Namizəd haqqında ətraflı analiz hesabatı hazırlayır
"""

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


def _call_openai(messages: list, max_tokens: int = 2000) -> Optional[str]:
    """OpenAI API-yə sorğu göndərir."""
    # Əvvəlcə config.py-dan, sonra environment variable-dan oxu
    api_key = ""
    try:
        from config import OPENAI_API_KEY
        api_key = OPENAI_API_KEY
    except ImportError:
        pass

    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        logger.error(
            "OPENAI_API_KEY tapılmadı!\n"
            "  config.py faylında OPENAI_API_KEY dəyərini yazın."
        )
        return None

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "messages": messages,
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

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error("OpenAI API xətası %d: %s", e.code, body[:300])
        return None
    except Exception as e:
        logger.error("OpenAI sorğu xətası: %s", e)
        return None


# ─── Dil aşkarlama və tərcümə ─────────────────────────────────────────────

def translate_if_needed(text: str) -> str:
    """
    Mətnin dilini yoxlayır, ingilis deyilsə tərcümə edir.
    İstənilən dildə işləyir: Azərbaycan, Türk, Rus, Alman, Fransız, və s.
    """
    # Çox qısa mətnləri yoxlama
    if len(text.strip()) < 30:
        return text

    sample = text[:800]

    # Dili yoxla
    answer = _call_openai([{
        "role": "user",
        "content": f"Is the following text in English? Reply ONLY 'yes' or 'no'.\n\nText:\n{sample}",
    }], max_tokens=10)

    if answer is None or answer.strip().lower().startswith("yes"):
        return text

    logger.info("Qeyri-ingilis mətn aşkarlandı. Tərcümə edilir...")

    # Hissələrə böl və tərcümə et
    max_chunk = 3000
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_chunk:
            if current:
                chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)

    translated_parts = []
    for i, chunk in enumerate(chunks):
        result = _call_openai([{
            "role": "user",
            "content": (
                "Translate the following text to English. Rules:\n"
                "1. Keep technical terms as they are (Python, SQL, AWS, etc.)\n"
                "2. Keep names, dates, numbers as they are\n"
                "3. Keep formatting and structure\n"
                "4. Return ONLY the translation\n\n"
                f"Text:\n{chunk}"
            ),
        }], max_tokens=4000)

        translated_parts.append(result if result else chunk)
        logger.info("  Hissə %d/%d tərcümə edildi", i + 1, len(chunks))

    final = "\n".join(translated_parts)
    logger.info("Tərcümə tamamlandı (%d → %d simvol)", len(text), len(final))
    return final


# ─── CV sahə təsnifatı ────────────────────────────────────────────────────

def classify_cv_domain(cv_text: str, available_domains: List[str]) -> str:
    """
    CV mətnini analiz edib namizədin əsas sahəsini müəyyən edir.
    OpenAI GPT-4o-mini istifadə edir.
    """
    domains_str = ", ".join(available_domains)

    prompt = f"""Analyze this CV/resume and determine the candidate's primary professional domain.

Available domains: {domains_str}

Rules:
1. Choose exactly ONE domain from the list above
2. Base your decision on: skills, experience, technologies, projects, education, job titles
3. Reply with ONLY the domain name, nothing else
4. If uncertain, choose the closest match. Use "Other" only as last resort.

CV Text:
{cv_text[:3000]}"""

    result = _call_openai([{"role": "user", "content": prompt}], max_tokens=50)

    if result:
        domain = result.strip().replace(" ", "_")
        # Dəqiq uyğunluq yoxla
        for d in available_domains:
            if d.lower() == domain.lower():
                return d
        # Qismən uyğunluq
        for d in available_domains:
            if d.lower() in domain.lower() or domain.lower() in d.lower():
                return d

    logger.warning("Sahə təyin edilə bilmədi, 'Other' seçildi")
    return "Other"


# ─── Vakansiya sahə təyini ────────────────────────────────────────────────

def classify_job_domain(job_text: str, available_domains: List[str]) -> str:
    """İş elanının hansı sahəyə aid olduğunu müəyyən edir."""
    domains_str = ", ".join(available_domains)

    prompt = f"""Analyze this job description and determine which professional domain it belongs to.

Available domains: {domains_str}

Rules:
1. Choose exactly ONE domain from the list above
2. Reply with ONLY the domain name, nothing else

Job Description:
{job_text[:3000]}"""

    result = _call_openai([{"role": "user", "content": prompt}], max_tokens=50)

    if result:
        domain = result.strip().replace(" ", "_")
        for d in available_domains:
            if d.lower() == domain.lower():
                return d
        for d in available_domains:
            if d.lower() in domain.lower() or domain.lower() in d.lower():
                return d

    logger.warning("Vakansiya sahəsi təyin edilə bilmədi")
    return "Other"


# ─── Namizəd analiz hesabatı ──────────────────────────────────────────────

def generate_candidate_report(
    cv_text: str,
    job_text: str,
    match_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
) -> str:
    """
    Hər namizəd üçün ətraflı AI analiz hesabatı hazırlayır.
    """
    prompt = f"""Analyze this candidate's CV against the job requirements and provide a structured report.

JOB REQUIREMENTS:
{job_text[:2000]}

CANDIDATE CV:
{cv_text[:2500]}

MATCH DATA:
- Match Score: {match_score:.1f}%
- Matched Skills: {', '.join(matched_skills[:15])}
- Missing Skills: {', '.join(missing_skills[:10])}

Provide a report in EXACTLY this format (fill in the details):

Candidate Name: [extract from CV]
Match Score: {match_score:.1f}%

Matched Skills:
[list each matched skill with a dash]

Relevant Experience:
[list relevant experience items]

Missing Requirements:
[list what's missing]

Strengths:
[list 2-3 key strengths]

Weaknesses:
[list 1-2 weaknesses]

Recommendation:
[2-3 sentence final recommendation]"""

    result = _call_openai([{"role": "user", "content": prompt}], max_tokens=1500)
    return result if result else f"Candidate analysis unavailable (Match Score: {match_score:.1f}%)"
