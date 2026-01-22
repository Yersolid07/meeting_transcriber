"""
Advanced NLP utilities: NER + dependency parsing wrapper with graceful fallbacks.

Provides a small abstraction `AdvancedNLPExtractor` that will use spaCy if available
(or fallback regex/heuristic extractors) to extract structured action items and
decisions from sentence-level metadata.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

try:
    import spacy
    from spacy.language import Language
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

try:
    from langdetect import detect as _detect_lang
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

logger = logging.getLogger("AdvancedNLP")


class AdvancedNLPExtractor:
    """Wrapper providing NER and dependency-based extraction.

    Usage:
        extractor = AdvancedNLPExtractor()
        items = extractor.extract_actions_from_sentences(sent_meta)

    `sent_meta` is a list of dicts produced by `BERTSummarizer._get_sentences_with_meta`
    where each dict contains at least `text`, `speaker_id`, `start`, `end`.
    """

    def __init__(self, lang: Optional[str] = None):
        self.lang = lang
        self._nlp: Optional[Language] = None
        if _HAS_SPACY:
            try:
                model = self._choose_model(lang)
                if model is not None:
                    self._nlp = spacy.load(model)
                    logger.info(f"Loaded spaCy model: {model}")
            except Exception as e:
                logger.warning(f"spaCy model load failed: {e}")
                self._nlp = None
        else:
            logger.debug("spaCy not available; using heuristic fallbacks")

    def _choose_model(self, lang: Optional[str]) -> Optional[str]:
        # Prefer language-specific small models if available
        if lang is None and _HAS_LANGDETECT:
            return None  # leave None to let caller decide based on text
        if lang == "id":
            return "id_core_news_sm"
        if lang == "en":
            return "en_core_web_sm"
        # Fall back to cross-lingual entity model if present
        return "xx_ent_wiki_sm"

    def _detect_lang(self, text: str) -> Optional[str]:
        if not _HAS_LANGDETECT:
            return None
        try:
            return _detect_lang(text)
        except Exception:
            return None

    def _get_doc(self, text: str):
        # If spaCy is loaded, use it. Otherwise return None.
        if self._nlp is None:
            # try to lazily pick a model based on language
            if _HAS_SPACY:
                lang = self._detect_lang(text)
                model = self._choose_model(lang)
                if model:
                    try:
                        self._nlp = spacy.load(model)
                        logger.info(f"Lazy-loaded spaCy model: {model}")
                    except Exception:
                        self._nlp = None
            return None
        try:
            return self._nlp(text)
        except Exception:
            return None

    def extract_persons(self, text: str) -> List[str]:
        doc = self._get_doc(text)
        if doc is None:
            # simple regex: capitalized words sequences
            names = re.findall(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})*)\b", text)
            return list(dict.fromkeys(names))
        persons = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "PER")]
        # preserve order, unique
        return list(dict.fromkeys(persons))

    def extract_actions_from_sentences(self, sent_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return candidate action items extracted from sentence metadata.

        Each returned dict contains: {owner, task, sentence_idx, confidence}
        """
        results: List[Dict[str, Any]] = []

        texts = [s["text"] for s in sent_meta]
        full = " ".join(texts[: max(1, min(10, len(texts)))])
        lang = self._detect_lang(full) if _HAS_LANGDETECT else None

        for i, s in enumerate(sent_meta):
            text = s.get("text", "").strip()
            if not text:
                continue

            # Quick keyword filter (language-agnostic): if no action words, skip
            if not re.search(r"\b(akan|harus|perlu|tolong|mohon|harap|deadline|target|tugas|follow up|tindak lanjut|siapkan|buat|bikin|saya|aku|kami|kita)\b", text, flags=re.IGNORECASE):
                # also check for English keywords
                if not re.search(r"\b(will|shall|must|please|assign|task|deadline|action item|follow up|todo)\b", text, flags=re.IGNORECASE):
                    continue

            doc = self._get_doc(text)
            owner: Optional[str] = None
            task: Optional[str] = None
            confidence = 0.5

            # First, try to find PERSON entities in the sentence
            if doc is not None:
                persons = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "PER")]
                if persons:
                    owner = persons[0]
                    confidence = 0.8

                # dependency parse-based task extraction
                try:
                    # find ROOT verb
                    root = None
                    for token in doc:
                        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                            root = token
                            break

                    if root is not None:
                        # look for direct objects / xcomp / ccomp
                        objs = [t for t in doc if t.dep_ in ("dobj", "obj", "xcomp", "ccomp")]
                        if objs:
                            task = " ".join([tok.text for tok in objs[0].subtree])
                            confidence = max(confidence, 0.7)
                        else:
                            # fallback: use root subtree as task
                            task = " ".join([tok.text for tok in root.subtree])
                            confidence = max(confidence, 0.6)

                    # If no owner found, search preceding tokens for personal pronouns
                    if owner is None:
                        pron = [t for t in doc if t.pos_ == "PRON"]
                        if pron:
                            owner = pron[0].text
                            confidence = 0.6
                except Exception:
                    pass

            # Regex fallback to capture "Name akan <action>" in many languages
            if owner is None:
                m = re.search(r"\b([A-Z][a-z]{1,20})\b\s+(akan|will|harus|must|to)\s+(?P<task>.+)", text, flags=re.IGNORECASE)
                if m:
                    owner = m.group(1)
                    task = m.group("task").strip(" .,:;-")
                    confidence = 0.7

            # Otherwise, check for "Saya akan"/"Aku akan" and attribute to speaker
            if owner is None and re.search(r"\b(saya|aku|kami|kita)\b", text, flags=re.IGNORECASE):
                owner = s.get("speaker_id")
                # try extract phrase after 'akan' or commit verb
                m2 = re.search(r"\b(?:akan|saya akan|aku akan|saya akan membuat|aku akan membuat|tolong|siapkan|buat|bikin)\b\s*(?P<task>.+)$", text, flags=re.IGNORECASE)
                if m2:
                    task = m2.group("task").strip(" .,:;-")
                    confidence = 0.7

            # final fallback: if sentence contains action keywords, use whole sentence
            if task is None:
                # trim connectors and filler
                t = re.sub(r"^(oke|ya|nah|baik)\b[:,-]*", "", text, flags=re.IGNORECASE).strip()
                task = t[:300]

            # Basic length filter
            if task and len(task.split()) < 3:
                continue

            results.append({"owner": owner or s.get("speaker_id"), "task": task, "sentence_idx": i, "confidence": confidence})

        return results


def extract_decisions_from_sentences(sent_meta: List[Dict[str, Any]]) -> List[str]:
    """Simple decision extraction: look for decision keywords and return cleaned contexts."""
    results: List[str] = []
    decision_kw = re.compile(r"\b(diputuskan|disepakati|kesimpulan|keputusan|sepakat|setuju|disetujui|putus|decided|decision)\b", flags=re.IGNORECASE)

    for i, s in enumerate(sent_meta):
        text = s.get("text", "").strip()
        if not text:
            continue
        if decision_kw.search(text):
            cleaned = re.sub(r"\[.*?\]", "", text)
            results.append(cleaned.strip())

    return results
