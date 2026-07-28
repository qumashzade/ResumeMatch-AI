"""
main.py — CV Matching System v2 — Tam avtomatik HR pipeline.

Əmrlər:
  1. fetch-emails    — Email-dən CV-ləri yüklə
  2. classify        — CV-ləri AI ilə sahələrə bölüşdür
  3. match           — Vakansiyaya uyğun namizədləri tap
  4. full-pipeline   — Hamısını ardıcıl işlət
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from modules.data_loader import load_cvs_from_folder, load_document, load_job_description
from modules.preprocessing import preprocess_document, preprocess_all
from modules.embedding import EmbeddingEngine
from modules.similarity import SimilarityRanker
from modules.ai_classifier import (
    translate_if_needed,
    classify_cv_domain,
    classify_job_domain,
    generate_candidate_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cv_matcher")


def print_banner():
    print("\n" + "═" * 72)
    print("  CV MATCHING SYSTEM v2 — Intelligent HR Pipeline")
    print("═" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# 1. EMAIL-DƏN CV YÜKLƏMƏSİ
# ═══════════════════════════════════════════════════════════════════════════

def cmd_fetch_emails():
    """Email hesabından CV-ləri yükləyir."""
    from modules.email_reader import fetch_cvs_from_email

    print("\n── EMAIL-DƏN CV YÜKLƏMƏSİ ──────────────────────────────────")

    download_folder = os.path.join(DATA_FOLDER, "downloaded_cvs")

    records = fetch_cvs_from_email(
        imap_server=EMAIL_CONFIG["imap_server"],
        imap_port=EMAIL_CONFIG["imap_port"],
        email_address=EMAIL_CONFIG["email"],
        password=EMAIL_CONFIG["password"],
        mailbox=EMAIL_CONFIG["mailbox"],
        search_criteria=EMAIL_CONFIG["search_criteria"],
        download_folder=download_folder,
        max_emails=EMAIL_CONFIG.get("max_emails", 0),
    )

    total = sum(len(r.attachments) for r in records)
    print(f"\n  {total} CV yükləndi → {download_folder}/")

    # Yüklənmiş CV-lərin siyahısı
    for r in records:
        for att in r.attachments:
            print(f"    ✓ {os.path.basename(att)} — {r.sender}")

    return download_folder


# ═══════════════════════════════════════════════════════════════════════════
# 2. CV-LƏRİN AI İLƏ TƏSNİFATLANDIRILMASI
# ═══════════════════════════════════════════════════════════════════════════

def _get_already_classified() -> set:
    """Artıq sahə qovluqlarına yerləşdirilmiş CV adlarını qaytarır."""
    classified = set()
    for domain in DOMAINS:
        folder = os.path.join(BASE_CV_FOLDER, domain)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, f)):
                    classified.add(f)
    return classified


def cmd_classify(source_folder: str = None, exclude_files: list = None):
    """CV-ləri AI ilə sahələrə görə təsnifatlandırır və qovluqlara yerləşdirir."""
    print("\n── CV TƏSNİFATLANDIRMA ──────────────────────────────────────")

    exclude_files = exclude_files or []

    # Artıq təsnifatlandırılmış CV-ləri tap
    already_done = _get_already_classified()
    if already_done:
        print(f"  {len(already_done)} CV artıq təsnifatlandırılıb (atlanacaq)")

    # CV-lərin olduğu qovluqları müəyyən et
    folders_to_scan = []

    if source_folder and os.path.isdir(source_folder):
        folders_to_scan.append(source_folder)
    else:
        downloaded = os.path.join(DATA_FOLDER, "downloaded_cvs")
        if os.path.isdir(downloaded) and os.listdir(downloaded):
            folders_to_scan.append(downloaded)

    # Bütün mənbələrdən CV-ləri yüklə
    all_cvs = []
    for folder in folders_to_scan:
        cvs = load_cvs_from_folder(folder)
        all_cvs.extend(cvs)

    # İstisna faylları çıxar
    if exclude_files:
        all_cvs = [cv for cv in all_cvs if cv.filename not in exclude_files]

    # Artıq təsnifatlandırılmış CV-ləri çıxar
    new_cvs = [cv for cv in all_cvs if cv.filename not in already_done]
    skipped = len(all_cvs) - len(new_cvs)

    if not new_cvs:
        if skipped > 0:
            print(f"  Bütün {skipped} CV artıq təsnifatlandırılıb. Yeni CV yoxdur.")
        else:
            print("  Heç bir CV tapılmadı!")
        return

    print(f"  {len(new_cvs)} yeni CV təsnifatlandırılacaq ({skipped} atlandı)...")

    # Yalnız yeni CV-ləri təsnifatlandır
    classified = 0
    for doc in new_cvs:
        # Tərcümə et (lazım olarsa)
        translated_text = translate_if_needed(doc.raw_text)

        # Sahəni müəyyən et
        domain = classify_cv_domain(translated_text, DOMAINS)

        # Qovluğu yarat
        domain_folder = os.path.join(BASE_CV_FOLDER, domain)
        os.makedirs(domain_folder, exist_ok=True)

        # CV-ni kopyala
        dest = os.path.join(domain_folder, doc.filename)
        if not os.path.exists(dest):
            shutil.copy2(doc.filepath, dest)
            classified += 1
            print(f"    ✓ {doc.filename} → {domain}/")
        else:
            print(f"    ○ {doc.filename} → {domain}/ (artıq var)")

    print(f"\n  {classified} CV təsnifatlandırıldı")

    # Sahə statistikası
    print("\n  Sahə statistikası:")
    for domain in DOMAINS:
        folder = os.path.join(BASE_CV_FOLDER, domain)
        if os.path.isdir(folder):
            count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            if count > 0:
                print(f"    {domain}: {count} CV")


# ═══════════════════════════════════════════════════════════════════════════
# 3. VAKANSİYAYA UYĞUN NAMİZƏDLƏRİN TAPILMASI
# ═══════════════════════════════════════════════════════════════════════════

def cmd_match(job_file: str = None, top_k: int = None):
    """Vakansiyaya uyğun namizədləri tapır, seçir və hesabat hazırlayır."""
    print_banner()

    job_file = job_file or JOB_DESCRIPTION_FILE
    top_k = top_k or TOP_K
    t0 = time.time()

    # ── 1. İş elanını yüklə və tərcümə et ─────────────────────────────
    print("\n── ADDIM 1: İş elanı yüklənir ──────────────────────────────")
    jd = load_job_description(job_file)
    jd_translated = translate_if_needed(jd.raw_text)
    logger.info("İş elanı yükləndi: %s", jd.filename)

    # ── 2. Vakansiya sahəsini müəyyən et ───────────────────────────────
    print("\n── ADDIM 2: Vakansiya sahəsi müəyyən edilir ────────────────")
    job_domain = classify_job_domain(jd_translated, DOMAINS)
    print(f"  Vakansiya sahəsi: {job_domain}")

    # ── 3. Uyğun qovluqdakı CV-ləri yüklə + Extra_CVs ─────────────────
    print("\n── ADDIM 3: CV-lər yüklənir ────────────────────────────────")
    domain_folder = os.path.join(BASE_CV_FOLDER, job_domain)

    cvs = []

    # Sahə qovluğundan yüklə
    if os.path.isdir(domain_folder) and os.listdir(domain_folder):
        domain_cvs = load_cvs_from_folder(domain_folder)
        cvs.extend(domain_cvs)
        print(f"  {job_domain}/ qovluğundan: {len(domain_cvs)} CV")
    else:
        print(f"  XƏBƏRDARLIQ: {domain_folder}/ qovluğu boşdur!")

    # Extra_CVs qovluğundan HƏMİŞƏ yüklə (sahə qovluğu boş olsa belə)
    os.makedirs(EXTRA_CV_FOLDER, exist_ok=True)
    if os.listdir(EXTRA_CV_FOLDER):
        extra_cvs = load_cvs_from_folder(EXTRA_CV_FOLDER)
        # Dublikatları yoxla (eyni fayl adı)
        existing_names = {cv.filename for cv in cvs}
        new_extras = [cv for cv in extra_cvs if cv.filename not in existing_names]
        cvs.extend(new_extras)
        print(f"  Extra_CVs/ qovluğundan: {len(new_extras)} əlavə CV")
    else:
        print(f"  Extra_CVs/ qovluğu boşdur (əlavə CV-lər bura atıla bilər)")

    # Heç bir CV tapılmadısa, bütün sahələri yoxla
    if not cvs:
        print("  Bütün sahə qovluqlarında axtarılır...")
        for domain in DOMAINS:
            folder = os.path.join(BASE_CV_FOLDER, domain)
            if os.path.isdir(folder):
                domain_cvs = load_cvs_from_folder(folder)
                cvs.extend(domain_cvs)

    if not cvs:
        print("  Heç bir CV tapılmadı! Əvvəlcə CV-ləri yükləyin.")
        return

    print(f"  CƏMİ: {len(cvs)} CV analiz ediləcək")

    # ── 4. Emal et ─────────────────────────────────────────────────────
    print("\n── ADDIM 4: CV-lər emal edilir ──────────────────────────────")
    cv_processed = preprocess_all(cvs, translate_fn=translate_if_needed)
    jd_processed = preprocess_document(jd.filename, jd_translated)
    logger.info("JD bacarıqlar: %s", ", ".join(jd_processed.extracted_skills))

    # ── 5. Embedding və reytinq ────────────────────────────────────────
    print("\n── ADDIM 5: Reytinqlənir ────────────────────────────────────")
    engine = EmbeddingEngine(mode=EMBEDDING_MODE)
    cv_texts = [cv.weighted_text for cv in cv_processed]

    if not cv_texts:
        print("  CV-lər emal edilə bilmədi!")
        return

    engine.fit_transform(cv_texts)

    ranker = SimilarityRanker()
    results = ranker.rank(
        embedding_engine=engine,
        jd_processed=jd_processed,
        cv_processed_list=cv_processed,
        top_k=top_k,
    )

    # ── 6. Top namizədləri seç və kopyala ──────────────────────────────
    print("\n── ADDIM 6: Top namizədlər seçilir ─────────────────────────")

    # Selected qovluğunu təmizlə (Windows permission xətalarını həll et)
    try:
        if os.path.exists(SELECTED_FOLDER):
            for f in os.listdir(SELECTED_FOLDER):
                fp = os.path.join(SELECTED_FOLDER, f)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass
    except Exception:
        pass
    os.makedirs(SELECTED_FOLDER, exist_ok=True)

    # filepath xəritəsi
    filepath_map = {doc.filename: doc for doc in cvs}

    print(f"\n  TOP {len(results)} NAMİZƏD")
    print("  " + "─" * 50)

    for rank, r in enumerate(results, 1):
        print(f"\n  #{rank}  {r.filename}")
        print(f"       Bal:       {r.final_score:.1f} / 100")
        print(f"       Bacarıqlar: {len(r.matched_skills)} uyğun, {len(r.missing_skills)} çatışmır")
        print(f"       → {r.explanation}")

        # Faylı kopyala
        doc = filepath_map.get(r.filename)
        if doc and os.path.isfile(doc.filepath):
            dest_name = f"{rank}_{r.filename}"
            shutil.copy2(doc.filepath, os.path.join(SELECTED_FOLDER, dest_name))

    print(f"\n  {len(results)} CV kopyalandı → {SELECTED_FOLDER}/")

    # ── 7. AI hesabat hazırla ──────────────────────────────────────────
    print("\n── ADDIM 7: AI analiz hesabatı hazırlanır ───────────────────")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  CV MATCHING — NAMİZƏD ANALİZ HESABATI")
    report_lines.append(f"  Tarix: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"  Vakansiya: {jd.filename}")
    report_lines.append(f"  Sahə: {job_domain}")
    report_lines.append(f"  Analiz edilən CV sayı: {len(cvs)}")
    report_lines.append(f"  Seçilən namizəd sayı: {len(results)}")
    report_lines.append("=" * 70)

    for rank, r in enumerate(results, 1):
        report_lines.append(f"\n{'─' * 70}")
        report_lines.append(f"  #{rank}")
        report_lines.append(f"{'─' * 70}\n")

        # AI ilə ətraflı analiz
        doc = filepath_map.get(r.filename)
        cv_text = doc.raw_text if doc else ""

        analysis = generate_candidate_report(
            cv_text=cv_text,
            job_text=jd.raw_text,
            match_score=r.final_score,
            matched_skills=r.matched_skills,
            missing_skills=r.missing_skills,
        )
        report_lines.append(analysis)
        print(f"  ✓ #{rank} analiz edildi")

    report_lines.append(f"\n{'=' * 70}")
    report_lines.append("  HESABAT SONU")
    report_lines.append(f"{'=' * 70}\n")

    # Hesabatı yaz
    report_text = "\n".join(report_lines)
    report_path = os.path.join(SELECTED_FOLDER, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  Hesabat yazıldı → {report_path}")

    # ── 8. JSON export (UI dashboard üçün) ─────────────────────────────
    import json

    # Sahə statistikası
    domain_stats = []
    for domain in DOMAINS:
        folder = os.path.join(BASE_CV_FOLDER, domain)
        count = 0
        if os.path.isdir(folder):
            count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        if count > 0:
            domain_stats.append({"name": domain, "count": count})

    # Namizəd nəticələri
    candidates = []
    for rank, r in enumerate(results, 1):
        candidates.append({
            "rank": rank,
            "filename": r.filename,
            "score": r.final_score,
            "embedding_score": r.embedding_score,
            "bm25_score": r.bm25_score,
            "skill_overlap_score": r.skill_overlap_score,
            "metadata_score": r.metadata_score,
            "matched_skills": r.matched_skills,
            "missing_skills": r.missing_skills,
            "explanation": r.explanation,
        })

    dashboard_data = {
        "job_domain": job_domain,
        "total_cvs_analyzed": len(cvs),
        "top_k": top_k,
        "domains": domain_stats,
        "candidates": candidates,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    json_path = os.path.join(SELECTED_FOLDER, "dashboard_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    print(f"  Dashboard data → {json_path}")

    # ── 9. Dashboard-u avtomatik aç ────────────────────────────────────
    import webbrowser

    dashboard_src = os.path.join(os.path.dirname(__file__), "dashboard.html")
    dashboard_dest = os.path.join(SELECTED_FOLDER, "dashboard.html")

    if os.path.isfile(dashboard_src):
        # HTML şablonunu oxu
        with open(dashboard_src, "r", encoding="utf-8") as f:
            html_template = f.read()

        # Report mətnini JavaScript-safe et
        report_js = report_text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

        # Data-nı JSON string-ə çevir
        json_str = json.dumps(dashboard_data, ensure_ascii=False)

        # Placeholder sətirini tap və data inject et
        data_script = f"// PIPELINE_DATA_INJECT placeholder — pipeline bu sətiri real data ilə əvəz edir\nDATA = {json_str};\nREPORT = `{report_js}`;"

        html_final = html_template.replace(
            "// PIPELINE_DATA_INJECT placeholder — pipeline bu sətiri real data ilə əvəz edir",
            data_script
        )

        with open(dashboard_dest, "w", encoding="utf-8") as f:
            f.write(html_final)

        # Brauzerdə aç
        abs_path = os.path.abspath(dashboard_dest)
        webbrowser.open(f"file:///{abs_path}")
        print(f"  Dashboard açıldı → {dashboard_dest}")
    else:
        print(f"  XƏBƏRDARLIQ: dashboard.html tapılmadı")

    elapsed = time.time() - t0
    print(f"\n  Pipeline {elapsed:.1f} saniyədə tamamlandı")
    print("═" * 72 + "\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. TAM PİPELİNE (hamısı birlikdə)
# ═══════════════════════════════════════════════════════════════════════════

def cmd_full_pipeline(job_file=None, top_k=None, skip_email=False):
    """Bütün addımları ardıcıl işlədir."""
    print_banner()

    download_folder = None
    if not skip_email:
        try:
            download_folder = cmd_fetch_emails()
        except Exception as e:
            logger.warning("Email yükləmə atlandı: %s", e)

    # Email-dən yüklənən CV-ləri classify et
    if download_folder:
        cmd_classify(source_folder=download_folder)

    # data/ qovluğundakı CV-ləri də classify et (job_description.txt istisna)
    if os.path.isdir(DATA_FOLDER):
        # Job description faylını müəyyən et ki, classify-da atlansın
        jd_file = os.path.basename(job_file or JOB_DESCRIPTION_FILE)
        cmd_classify(source_folder=DATA_FOLDER, exclude_files=[jd_file])

    cmd_match(job_file=job_file, top_k=top_k)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CV Matching System v2 — Tam avtomatik HR pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Əmrlər")

    # fetch-emails
    sub_fetch = subparsers.add_parser("fetch-emails", help="Email-dən CV yüklə")

    # classify
    sub_classify = subparsers.add_parser("classify", help="CV-ləri sahələrə böl")
    sub_classify.add_argument(
        "--source", default=None,
        help="CV-lərin olduğu qovluq (default: data/downloaded_cvs və ya CVs/)"
    )

    # match
    sub_match = subparsers.add_parser("match", help="Vakansiyaya uyğun CV tap")
    sub_match.add_argument("--job-file", default=None, help="İş elanı faylı")
    sub_match.add_argument("--top-k", type=int, default=None, help="Neçə namizəd seçsin")

    # full-pipeline
    sub_full = subparsers.add_parser("full-pipeline", help="Hamısını işlət")
    sub_full.add_argument("--job-file", default=None, help="İş elanı faylı")
    sub_full.add_argument("--top-k", type=int, default=None, help="Neçə namizəd seçsin")
    sub_full.add_argument("--skip-email", action="store_true", help="Email yükləməni atla")

    # watch
    sub_watch = subparsers.add_parser("watch", help="Qovluqları izlə, yeni CV-ləri avtomatik təsnifatlandır")
    sub_watch.add_argument(
        "--folder", default=None,
        help="Əlavə izlənəcək qovluq (default: Extra_CVs/ və data/downloaded_cvs/)"
    )

    args = parser.parse_args()

    if args.command == "fetch-emails":
        cmd_fetch_emails()
    elif args.command == "classify":
        cmd_classify(source_folder=args.source)
    elif args.command == "match":
        cmd_match(job_file=args.job_file, top_k=args.top_k)
    elif args.command == "full-pipeline":
        cmd_full_pipeline(
            job_file=args.job_file,
            top_k=args.top_k,
            skip_email=args.skip_email,
        )
    elif args.command == "watch":
        from modules.watcher import start_watching
        watch_folders = [EXTRA_CV_FOLDER, os.path.join(DATA_FOLDER, "downloaded_cvs")]
        if args.folder:
            watch_folders.append(args.folder)
        print_banner()
        print("\n── WATCH REJİMİ ─────────────────────────────────────────────")
        start_watching(watch_folders, BASE_CV_FOLDER, DOMAINS)
    else:
        parser.print_help()
        print("\n  Nümunələr:")
        print("    python main.py classify --source data/my_cvs")
        print("    python main.py match --job-file data/job_description.txt --top-k 5")
        print("    python main.py full-pipeline --skip-email --top-k 3")
        print("    python main.py watch")
        print("    python main.py watch --folder C:\\Users\\me\\Downloads")


if __name__ == "__main__":
    main()
