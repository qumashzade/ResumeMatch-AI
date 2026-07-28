"""
similarity.py — Compute, combine, and rank CV-to-job similarity scores.

Implements a multi-signal ranking approach:
  1. Embedding similarity (semantic/TF-IDF from embedding engine)
  2. BM25 keyword matching (captures exact term overlap)
  3. Skill overlap bonus (direct skill matching)
  4. Education/experience metadata boost

Final score = weighted combination of all signals, producing an
interpretable 0–100 relevance score.
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ─── Score weights for final ranking ───────────────────────────────────────
DEFAULT_WEIGHTS = {
    "embedding": 0.45,    # semantic/tfidf similarity
    "bm25": 0.20,         # keyword overlap
    "skill_match": 0.25,  # direct skill overlap ratio
    "metadata": 0.10,     # education + experience bonus
}


@dataclass
class MatchResult:
    """Result for a single CV match."""
    filename: str
    final_score: float              # 0-100 composite score
    embedding_score: float          # raw cosine similarity
    bm25_score: float               # normalized BM25
    skill_overlap_score: float      # fraction of JD skills matched
    metadata_score: float           # education + experience bonus
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    explanation: str = ""


class SimilarityRanker:
    """
    Ranks CVs against a job description using multiple signals.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.bm25: BM25Okapi = None

    def _build_bm25_index(self, cv_tokens: List[List[str]]) -> None:
        """Build BM25 index from tokenized CV documents."""
        self.bm25 = BM25Okapi(cv_tokens)
        logger.info("BM25 index built with %d documents", len(cv_tokens))

    def _compute_bm25_scores(self, query_tokens: List[str]) -> np.ndarray:
        """Get BM25 scores for a query against all indexed documents."""
        raw_scores = self.bm25.get_scores(query_tokens)
        # Normalize to [0, 1], handle edge cases
        min_score = raw_scores.min()
        if min_score < 0:
            raw_scores = raw_scores - min_score  # shift to non-negative
        max_score = raw_scores.max() if raw_scores.max() > 0 else 1.0
        return raw_scores / max_score

    def _compute_skill_overlap(
        self, jd_skills: List[str], cv_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Compute skill overlap ratio and return matched/missing skills.
        """
        jd_set = set(s.lower() for s in jd_skills)
        cv_set = set(s.lower() for s in cv_skills)

        if not jd_set:
            return 0.0, [], []

        matched = sorted(jd_set & cv_set)
        missing = sorted(jd_set - cv_set)
        score = len(matched) / len(jd_set)
        return score, matched, missing

    def _compute_metadata_score(
        self, edu_score: int, years_exp: float
    ) -> float:
        """
        Compute a 0-1 metadata bonus based on education level and experience.
        For junior roles: Master's = ideal, some experience preferred.
        """
        # Education component (0-0.5)
        edu_component = min(edu_score / 4.0, 1.0) * 0.5

        # Experience component (0-0.5)
        # For junior role: 0-3 years is ideal, diminishing returns after
        exp_component = min(years_exp / 4.0, 1.0) * 0.5

        return edu_component + exp_component

    def _generate_explanation(self, result: MatchResult) -> str:
        """Generate a human-readable explanation of why a CV matched."""
        parts = []

        # Score tier
        if result.final_score >= 80:
            parts.append("Excellent match.")
        elif result.final_score >= 65:
            parts.append("Strong match.")
        elif result.final_score >= 50:
            parts.append("Moderate match.")
        else:
            parts.append("Partial match.")

        # Skills
        n_matched = len(result.matched_skills)
        n_missing = len(result.missing_skills)
        if n_matched > 0:
            top_skills = result.matched_skills[:8]
            parts.append(f"Matches {n_matched} key skills: {', '.join(top_skills)}.")
        if n_missing > 0 and n_missing <= 5:
            parts.append(f"Missing: {', '.join(result.missing_skills[:5])}.")

        # Semantic
        if result.embedding_score > 0.6:
            parts.append("Strong semantic alignment with job requirements.")
        elif result.embedding_score > 0.4:
            parts.append("Good semantic relevance to the role.")

        return " ".join(parts)

    def rank(
        self,
        embedding_engine,
        jd_processed,
        cv_processed_list: list,
        top_k: int = 5,
    ) -> List[MatchResult]:
        """
        Full ranking pipeline:
          1. Embedding similarity via the engine
          2. BM25 keyword scores
          3. Skill overlap
          4. Metadata bonuses
          5. Weighted combination → sorted results

        Returns top_k MatchResult objects.
        """
        n = len(cv_processed_list)
        logger.info("Ranking %d CVs against job description", n)

        # ── 1. Embedding similarity ────────────────────────────────────
        query_vec = embedding_engine.transform_query(jd_processed.weighted_text)
        emb_scores, emb_indices = embedding_engine.search_faiss(query_vec, top_k=n)

        # Map index → score
        emb_score_map = {}
        for score, idx in zip(emb_scores, emb_indices):
            if idx >= 0:  # FAISS may return -1 for padding
                emb_score_map[idx] = float(score)

        # ── 2. BM25 scores ─────────────────────────────────────────────
        cv_token_lists = [cv.skill_tokens for cv in cv_processed_list]
        self._build_bm25_index(cv_token_lists)
        bm25_scores = self._compute_bm25_scores(jd_processed.skill_tokens)

        # ── 3 & 4. Per-CV scoring ──────────────────────────────────────
        results: List[MatchResult] = []
        jd_skills = jd_processed.extracted_skills

        for i, cv in enumerate(cv_processed_list):
            e_score = emb_score_map.get(i, 0.0)
            b_score = float(bm25_scores[i])
            s_score, matched, missing = self._compute_skill_overlap(
                jd_skills, cv.extracted_skills
            )
            m_score = self._compute_metadata_score(
                cv.education_score, cv.years_experience
            )

            # Weighted combination → scale to 0-100
            composite = (
                self.weights["embedding"] * e_score
                + self.weights["bm25"] * b_score
                + self.weights["skill_match"] * s_score
                + self.weights["metadata"] * m_score
            ) * 100

            result = MatchResult(
                filename=cv.filename,
                final_score=round(composite, 2),
                embedding_score=round(e_score, 4),
                bm25_score=round(b_score, 4),
                skill_overlap_score=round(s_score, 4),
                metadata_score=round(m_score, 4),
                matched_skills=matched,
                missing_skills=missing,
            )
            result.explanation = self._generate_explanation(result)
            results.append(result)

        # Sort descending by final_score
        results.sort(key=lambda r: r.final_score, reverse=True)

        logger.info(
            "Top match: %s (%.1f), Bottom: %s (%.1f)",
            results[0].filename, results[0].final_score,
            results[-1].filename, results[-1].final_score,
        )

        return results[:top_k]
