"""
BERT Extractive Summarization Module
====================================
Implements extractive summarization using IndoBERT/mBERT for meeting minutes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _collapse_repeated_phrases_global(text: str, max_ngram: int = 6, min_repeats: int = 2) -> str:
    """Module-level helper to collapse repeated n-gram phrases.

    Iteratively collapses repeated adjacent n-gram phrases into a single occurrence.
    """
    if not text or min_repeats < 2:
        return text
    pattern = re.compile(r"(\b(?:\w+\s+){0,%d}\w+\b)(?:\s+\1){%d,}" % (max_ngram - 1, min_repeats - 1), flags=re.IGNORECASE)
    prev = None
    out = text
    while prev != out:
        prev = out
        out = pattern.sub(r"\1", out)
    return out

from src.transcriber import TranscriptSegment


@dataclass
class SummarizationConfig:
    """Configuration for summarization"""

    # Method: 'extractive' (BERT embeddings) or 'abstractive' (seq2seq model)
    method: str = "extractive"

    # Models
    # Use a cached/available model for reliability in offline environments
    sentence_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    abstractive_model_id: str = "google/mt5-base"

    # Extractive settings (increase to capture more key points)
    num_sentences: int = 7
    min_sentence_length: int = 6
    max_sentence_length: int = 300

    # Abstractive settings
    max_input_chars: int = 1000
    max_summary_length: int = 128
    min_summary_length: int = 30

    # Light abstractive refinement step (run on condensed extractive overview)
    do_abstractive_refinement: bool = True
    abstractive_refine_max_len: int = 80

    # Generate a comprehensive executive overview (long, covering entire meeting)
    comprehensive_overview: bool = True
    comprehensive_max_length: int = 512

    # Post-processing options
    polish_overview: bool = True
    semantic_dedup_threshold: float = 0.75

    # Scoring weights
    position_weight: float = 0.15
    length_weight: float = 0.10
    similarity_weight: float = 0.75

    # Keywords for detection
    decision_keywords: List[str] = field(
        default_factory=lambda: [
            "diputuskan",
            "disepakati",
            "kesimpulan",
            "keputusan",
            "jadi",
            "maka",
            "sepakat",
            "setuju",
            "final",
            "kesepakatan",
            "disimpulkan",
            "ditetapkan",
            "disetujui",
            "putus",
        ]
    )

    action_keywords: List[str] = field(
        default_factory=lambda: [
            "akan",
            "harus",
            "perlu",
            "tolong",
            "mohon",
            "harap",
            "deadline",
            "target",
            "tugas",
            "tanggung jawab",
            "action item",
            "follow up",
            "tindak lanjut",
            "dikerjakan",
            "selesaikan",
            "lakukan",
            "siapkan",
            "minggu depan",
            "besok",
            "segera",
            "bikin",
            "buat",
        ]
    )

    # Device
    device: str = "cpu"


@dataclass
class MeetingSummary:
    """Structured meeting summary"""

    overview: str
    key_points: List[str]
    decisions: List[str]
    action_items: List[Dict[str, str]]
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overview": self.overview,
            "key_points": self.key_points,
            "decisions": self.decisions,
            "action_items": self.action_items,
            "topics": self.topics,
            "keywords": getattr(self, "keywords", []),
        }

    def __str__(self) -> str:
        """String representation"""
        lines = []
        lines.append("=== RINGKASAN RAPAT ===\n")
        lines.append(f"Overview:\n{self.overview}\n")

        if self.key_points:
            lines.append("Poin-Poin Penting:")
            for i, point in enumerate(self.key_points, 1):
                lines.append(f"  {i}. {point}")
            lines.append("")

        if self.decisions:
            lines.append("Keputusan:")
            for i, decision in enumerate(self.decisions, 1):
                lines.append(f"  {i}. {decision}")
            lines.append("")

        if self.action_items:
            lines.append("Action Items:")
            for i, item in enumerate(self.action_items, 1):
                owner = item.get("owner", "TBD")
                task = item.get("task", "")
                due = item.get("due", "")
                if due:
                    lines.append(f"  {i}. [{owner}] {task} (Due: {due})")
                else:
                    lines.append(f"  {i}. [{owner}] {task}")

        if self.topics:
            lines.append("")
            lines.append("Topik:")
            lines.append(", ".join(self.topics))

        return "\n".join(lines)

    def to_json(self) -> str:
        """Return a JSON string for machine-readable outputs."""
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_yaml(self) -> str:
        """Return a YAML string (requires PyYAML)."""
        try:
            import yaml

            return yaml.safe_dump(self.to_dict(), allow_unicode=True)
        except Exception:
            # Fallback to JSON if YAML not available
            return self.to_json()


class AbstractiveSummarizer:
    """Abstractive summarizer using HuggingFace transformers pipeline (mt5/mbart/etc)."""

    def __init__(self, config: Optional[SummarizationConfig] = None):
        self.config = config or SummarizationConfig()
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline

                device = 0 if self.config.device.startswith("cuda") else -1
                print(f"[Summarizer] Loading abstractive model: {self.config.abstractive_model_id}")
                self._pipeline = pipeline(
                    "summarization",
                    model=self.config.abstractive_model_id,
                    tokenizer=self.config.abstractive_model_id,
                    device=device,
                    truncation=True,
                )
                print("[Summarizer] Abstractive model loaded successfully")
            except Exception as e:
                print(f"[Summarizer] Warning: abstractive model load failed: {e}")
                self._pipeline = None

    def _chunk_text(self, text: str) -> List[str]:
        max_chars = int(self.config.max_input_chars)
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            # try to cut at sentence boundary
            cut = text.rfind(".", start, end)
            if cut <= start:
                cut = end
            chunk = text[start:cut].strip()
            if chunk:
                # prevent repeating identical chunks
                chunk = self._collapse_repeated_phrases(chunk)
                chunks.append(chunk)
            start = cut
        return chunks

    def _clean_abstractive_output(self, overview: str, full_text: str) -> (str, List[str]):
        """Clean artifacts from abstractive model output and produce fallback key points.

        Returns (overview_clean, key_points)
        """
        overview_clean = self._clean_abstractive_text(overview)

        # If abstract output is still noisy (placeholders remain or too few alpha tokens), fallback to extractive
        if "<extra_id" in overview or len(re.findall(r"[a-zA-Z]{2,}", overview_clean)) < 10 or re.search(r"\b(\w+)(?:\s+\1){2,}", overview_clean.lower()):
            sentences = BERTSummarizer(self.config)._split_sentences(full_text)
            key_points = [s for s in sentences[: self.config.num_sentences]]
            overview_clean = " ".join(key_points[:3])
            return overview_clean, key_points

        # Otherwise make sure key points are meaningful and deduplicated
        parts = [s.strip() for s in re.split(r"\.|!|\?", overview_clean) if s.strip()]
        seen_kp = set()
        key_points: List[str] = []
        for p in parts:
            p_clean = re.sub(r"[^\w\s]", "", p) if p else p
            p_clean = re.sub(r"\s+", " ", p_clean).strip()
            if len(p_clean.split()) < 3:
                continue
            low = p_clean.lower()
            if low in seen_kp:
                continue
            seen_kp.add(low)
            key_points.append(p_clean)
            if len(key_points) >= self.config.num_sentences:
                break

        return overview_clean, key_points

    def _clean_abstractive_text(self, text: str) -> str:
        """Lightweight cleaning of abstractive text outputs (remove placeholders, collapse punctuation).

        Kept as a separate method for unit testing/backwards compatibility with older tests.
        Also collapses repeated trivial tokens and reduces punctuation runs.
        """
        t = re.sub(r"<extra_id_\d+>", "", text)
        t = re.sub(r"\)\s*<extra_id_\d+>", "", t)
        # collapse repeated short filler words sequences e.g. "Jadi contohnya Jadi contohnya ..."
        t = self._collapse_repeated_phrases(t)
        t = re.sub(r"\s*[\.]{2,}\s*", ". ", t)
        t = re.sub(r"[!?]{2,}", ".", t)
        t = re.sub(r"\s+", " ", t).strip()
        # Remove leading/trailing hyphens and stray punctuation
        t = re.sub(r"^[-\s]+|[-\s]+$", "", t)
        if not re.search(r"[.!?]$", t):
            t = t + "."
        return t

    def _generate_keywords(self, text: str, top_k: int = 8) -> List[str]:
        """Generate simple keywords by frequency (fallback)."""
        toks = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        freq = {}
        stop = {"yang","dan","ini","itu","untuk","dengan","juga","sudah","ada","kita","saya","kamu"}
        for w in toks:
            if w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_k]]

    def _collapse_repeated_phrases(self, text: str, max_ngram: int = 6, min_repeats: int = 2) -> str:
        """Delegates to module-level collapse helper"""
        return _collapse_repeated_phrases_global(text, max_ngram=max_ngram, min_repeats=min_repeats)

    def _semantic_deduplicate(self, items: List[str], threshold: Optional[float] = None) -> List[str]:
        """Delegate to AbstractiveSummarizer's semantic dedupe for compatibility."""
        return AbstractiveSummarizer(self.config)._semantic_deduplicate(items, threshold)

    def _semantic_dedup_action_items(self, actions: List[Dict[str, str]], threshold: Optional[float] = None) -> List[Dict[str, str]]:
        """Delegate to AbstractiveSummarizer's action-item dedupe for compatibility."""
        return AbstractiveSummarizer(self.config)._semantic_dedup_action_items(actions, threshold)

    def _parse_structured_output(self, raw: str, defaults: Dict[str, Any]) -> (str, List[str]):
        """Try to parse YAML/JSON or simple structured text into (overview, keywords).

        If parsing fails, return (cleaned_raw, fallback_keywords)
        """
        cleaned = raw.strip()

        # Try YAML first (if available)
        try:
            import yaml

            parsed = yaml.safe_load(cleaned)
            if isinstance(parsed, dict):
                ov = parsed.get("overview", "")
                kws = parsed.get("keywords", None)
                if kws is None:
                    kws = self._generate_keywords(ov or " ".join(defaults.get("key_points", [])))
                return (ov.strip() if isinstance(ov, str) else "", kws)
        except Exception:
            pass

        # Try JSON
        try:
            import json

            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                ov = parsed.get("overview", "")
                kws = parsed.get("keywords", None)
                if kws is None:
                    kws = self._generate_keywords(ov or " ".join(defaults.get("key_points", [])))
                return (ov.strip() if isinstance(ov, str) else "", kws)
        except Exception:
            pass

        # Simple heuristic: look for header 'overview:' or 'Ringkasan:' in text
        m = re.search(r"(?im)^(overview|ringkasan)\s*:\s*(.*)$", cleaned)
        if m:
            ov = m.group(2).strip()
            kws = self._generate_keywords(ov or " ".join(defaults.get("key_points", [])))
            return ov, kws

        # If nothing recognized, return fallback cleaned text and keywords
        return cleaned, self._generate_keywords(cleaned or " ".join(defaults.get("key_points", [])))

    def _sanitize_for_prompt(self, text: str) -> str:
        """Sanitize text before injecting into the prompt: remove model placeholders, URLs/domains/emails,
        common web-article boilerplate (closing lines like "Semoga bermanfaat"), and collapse repeats."""
        if not text:
            return text
        t = re.sub(r"<extra_id_\d+>", "", text)
        # remove emails
        t = re.sub(r"\b\S+@\S+\.\S+\b", " ", t)
        # remove domain-like tokens (e.g., Eksekutif.com.co.id)
        t = re.sub(r"\b\S+\.(?:com|co\.id|info|id|net|org)(?:\.[a-z]{2,})*\b", " ", t, flags=re.IGNORECASE)
        # remove common article/web boilerplate short phrases that often appear as closings
        t = re.sub(r"(?i)\b(semoga artikel ini bermanfaat(?: bagi anda semua)?|semoga bermanfaat|terima kasih(?: atas masukannya| juga)?)\b[.!\s,]*", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = _collapse_repeated_phrases_global(t)
        return t

    def _is_repetitive_text(self, text: str, max_run: int = 6) -> bool:
        """Detect highly repetitive model outputs (including repeated n-gram phrases).

        Returns True if repetition patterns exceed thresholds.
        """
        if not text:
            return False
        # check placeholder presence quickly
        if re.search(r"<extra_id_\d+>", text):
            return True
        # Tokenize
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return False
        # Check simple token runs
        run = 1
        last = tokens[0]
        for tok in tokens[1:]:
            if tok == last:
                run += 1
                if run >= max_run:
                    return True
            else:
                last = tok
                run = 1
        # Check n-gram repeated phrase runs for n=1..4
        max_ngram = 4
        n_tokens = len(tokens)
        for n in range(1, max_ngram + 1):
            i = 0
            while i + 2 * n <= n_tokens:
                # compare tokens[i:i+n] with subsequent repeated occurrences
                pattern = tokens[i:i + n]
                run = 1
                j = i + n
                while j + n <= n_tokens and tokens[j:j + n] == pattern:
                    run += 1
                    j += n
                    if run >= max_run:
                        return True
                i += 1
        # fallback regex for single-token repetition
        if re.search(r"(\b\w+\b)(?:\s+\1\b){%d,}" % (max_run - 1), text.lower()):
            return True
        return False

    def _contains_domain_noise(self, text: str) -> bool:
        """Detect domain-like or short web boilerplate noise (e.g., 'Eksekutif.com', 'Semoga artikel ini bermanfaat').

        Returns True if common domain patterns or boilerplate phrases are found.
        """
        if not text:
            return False
        if re.search(r"\b\S+\.(?:com|co\.id|info|id|net|org)(?:\.[a-z]{2,})*\b", text, flags=re.IGNORECASE):
            return True
        if re.search(r"(?i)\b(semoga artikel ini bermanfaat(?: bagi anda semua)?|semoga bermanfaat|terima kasih)\b", text):
            return True
        return False

    def _normalize_overview_text(self, text: str) -> str:
        """Normalize overview into a readable paragraph or keep structured lists tidy."""
        if not text:
            return text
        t = text.strip()
        # collapse repeated fragments first
        t = _collapse_repeated_phrases_global(t)

        # If text contains list markers or section headers, tidy spacing and return
        if "\n-" in t or "Poin-Poin Penting" in t or "Keputusan" in t or "Action Items" in t:
            # normalize newlines and strip extra spaces
            t = re.sub(r"\n\s+", "\n", t)
            t = re.sub(r"\n{2,}", "\n\n", t)
            return t.strip()

        # Otherwise make a single paragraph and deduplicate near-duplicate fragments
        # split by common separators (newline, bullet, or hyphen sequences)
        if " - " in t:
            parts = [p.strip(" -" ) for p in re.split(r"\s*-\s*", t) if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]

        seen = set()
        uniq = []
        for p in parts:
            norm = re.sub(r"[^a-z0-9 ]", "", p.lower())
            norm = re.sub(r"\s+", " ", norm).strip()
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            uniq.append(p.strip(" -."))

        para = " ".join(uniq)
        para = re.sub(r"\s+", " ", para).strip()

        # Remove any leftover emails/domains or short web boilerplate that slipped through
        para = re.sub(r"\b\S+@\S+\.\S+\b", " ", para)
        para = re.sub(r"\b\S+\.(?:com|co\.id|info|id|net|org)(?:\.[a-z]{2,})*\b", " ", para, flags=re.IGNORECASE)
        para = re.sub(r"(?i)\b(semoga artikel ini bermanfaat(?: bagi anda semua)?|semoga bermanfaat|terima kasih(?: atas masukannya| juga)?)\b[.!\s,]*", " ", para)
        para = re.sub(r"\s+", " ", para).strip()

        if para and not re.search(r"[.!?]$", para):
            para = para + "."
        if para:
            para = para[0].upper() + para[1:]
        return para

    def _polish_overview(self, overview: str, full_text: str) -> str:
        """Polish overview into an executive, coherent paragraph using abstractive model (if available).

        Falls back to normalization and deduplication if model not available.
        """
        if not overview:
            return overview
        # Basic normalization first
        overview = _collapse_repeated_phrases_global(overview)
        overview = self._normalize_overview_text(overview)

        # If model available and config allows, ask for paraphrase/expansion
        if getattr(self.config, "polish_overview", True):
            try:
                self._load_model()
                if self._pipeline is not None:
                    prompt = (
                        "Paraphrase dan perluas teks berikut menjadi paragraf eksekutif yang jelas, ringkas, dan mudah dibaca. "
                        "Jangan sertakan header."
                        "\n\nTeks:\n" + overview
                    )
                    out = self._pipeline(
                        prompt,
                        max_length=min(getattr(self.config, "comprehensive_max_length", 512), 350),
                        min_length=40,
                        truncation=True,
                        do_sample=False,
                    )
                    if isinstance(out, list) and out:
                        candidate = out[0].get("summary_text", "").strip()
                        candidate = self._clean_abstractive_text(candidate)
                        candidate = _collapse_repeated_phrases_global(candidate)
                        candidate = self._normalize_overview_text(candidate)
                        return candidate
            except Exception:
                pass

        return overview

    def _semantic_deduplicate(self, items: List[str], threshold: Optional[float] = None) -> List[str]:
        """Deduplicate similar items using sentence-transformer embeddings + cosine similarity.

        Returns the first occurrence for each semantic group.
        """
        if not items:
            return []
        thr = threshold if threshold is not None else getattr(self.config, "semantic_dedup_threshold", 0.75)
        # try embeddings
        try:
            embs = self._compute_embeddings(items)
            if embs is not None:
                from sklearn.metrics.pairwise import cosine_similarity

                sim = cosine_similarity(embs)
                n = len(items)
                taken = set()
                result = []
                for i in range(n):
                    if i in taken:
                        continue
                    result.append(items[i])
                    for j in range(i + 1, n):
                        if sim[i, j] >= thr:
                            taken.add(j)
                # If embeddings didn't merge anything useful, fallback to token-jaccard grouping
                if len(result) == len(items) and len(items) > 1:
                    # token Jaccard
                    token_sets = [set(re.findall(r"\w+", it.lower())) for it in items]
                    taken2 = set()
                    result2 = []
                    for i in range(len(items)):
                        if i in taken2:
                            continue
                        result2.append(items[i])
                        for j in range(i + 1, len(items)):
                            if j in taken2:
                                continue
                            si = token_sets[i]
                            sj = token_sets[j]
                            if not si or not sj:
                                continue
                            jacc = len(si & sj) / float(len(si | sj))
                            if jacc >= 0.45:
                                taken2.add(j)
                    return result2
                return result
            else:
                raise ValueError("No embeddings")
        except Exception:
            # fallback to token-jaccard grouping first (robust when embeddings aren't available)
            try:
                token_sets = [set(re.findall(r"\w+", it.lower())) for it in items]
                taken = set()
                res = []
                for i in range(len(items)):
                    if i in taken:
                        continue
                    res.append(items[i])
                    si = token_sets[i]
                    for j in range(i + 1, len(items)):
                        if j in taken:
                            continue
                        sj = token_sets[j]
                        if not si or not sj:
                            continue
                        jacc = len(si & sj) / float(len(si | sj))
                        if jacc >= 0.45:
                            taken.add(j)
                return res
            except Exception:
                # final fallback to naive textual deduplication
                seen = set()
                res = []
                for it in items:
                    low = re.sub(r"\s+", " ", it.lower()).strip()
                    if low in seen:
                        continue
                    seen.add(low)
                    res.append(it)
                return res

    def _semantic_dedup_action_items(self, actions: List[Dict[str, str]], threshold: Optional[float] = None) -> List[Dict[str, str]]:
        """Deduplicate action items by task text; merge owners when necessary."""
        if not actions:
            return []
        tasks = [a.get("task", "") for a in actions]
        groups = self._semantic_deduplicate(tasks, threshold=threshold)
        # groups contains first representative tasks; now build merged items
        merged = []
        for rep in groups:
            owners = []
            timestamps = []
            dues = set()
            for a in actions:
                if a.get("task", "") == rep or (rep and rep in a.get("task", "")):
                    if a.get("owner") and a.get("owner") not in owners:
                        owners.append(a.get("owner"))
                    if a.get("timestamp"):
                        timestamps.append(a.get("timestamp"))
                    if a.get("due"):
                        dues.add(a.get("due"))
            owner_str = " / ".join(owners) if owners else "TBD"
            merged.append({
                "owner": owner_str,
                "task": rep,
                "timestamp": timestamps[0] if timestamps else "",
                "due": ", ".join(sorted(list(dues))) if dues else "",
            })
        return merged

    def generate_comprehensive_summary(self, full_text: str, key_points: List[str], decisions: List[str], action_items: List[Dict[str, str]], topics: List[str]) -> (str, List[str]):
        """Generate a comprehensive executive summary covering the meeting.

        Uses the abstractive pipeline with a guided prompt built from extracted components.
        Attempts to request YAML-structured output for reliable parsing; falls back to rule-based assembly.
        Returns (overview_text, keywords)
        """
        # Build a structured prompt that requests YAML output for safe parsing
        prompt_parts = [
            "Anda adalah asisten yang menulis ringkasan rapat yang komprehensif dan terstruktur.",
            "Output harus dalam format YAML dengan kunci: overview, key_points (list), decisions (list), action_items (list of {owner, task, due}), keywords (list).",
            "Berikan overview naratif yang jelas, serta daftar poin penting, keputusan, dan tindak lanjut.",
            "Topik yang dibahas:",
            ", ".join(topics) if topics else "-",
            "Poin-poin penting:\n" + "\n".join([f"- {p}" for p in key_points]) if key_points else "",
            "Keputusan:\n" + "\n".join([f"- {d}" for d in decisions]) if decisions else "",
            "Tindak lanjut (Action Items):\n" + "\n".join([f"- [{a.get('owner','TBD')}] {a.get('task','')}" for a in action_items]) if action_items else "",
            "Tuliskan field 'overview' minimal 80 kata sebagai paragraf naratif yang merangkum seluruh rapat dengan jelas.",
            "Mohon hasilkan YAML yang valid."
        ]
        prompt = "\n\n".join([p for p in prompt_parts if p])

        # Sanitize inputs to avoid placeholder tokens and repeated garbage
        key_points = [self._sanitize_for_prompt(k) for k in key_points if k and k.strip()]
        decisions = [self._sanitize_for_prompt(d) for d in decisions if d and d.strip()]
        for a in action_items:
            a['task'] = self._sanitize_for_prompt(a.get('task',''))

        # Deduplicate before sending to model
        try:
            key_points = self._semantic_deduplicate(key_points)
            decisions = self._semantic_deduplicate(decisions)
        except Exception:
            key_points = list(dict.fromkeys(key_points))
            decisions = list(dict.fromkeys(decisions))

        # Use pipeline if available
        try:
            self._load_model()
            if self._pipeline is not None:
                # Try up to 2 attempts: first deterministic, second sampled if repetition/shortness detected
                attempts = 2
                for attempt in range(attempts):
                    gen_kwargs = dict(
                        max_length=getattr(self.config, "comprehensive_max_length", 512),
                        min_length=max(80, int(getattr(self.config, "comprehensive_max_length", 512) * 0.12)),
                        truncation=True,
                        do_sample=False,
                        no_repeat_ngram_size=4,
                        repetition_penalty=1.3,
                    )
                    if attempt == 1:
                        # more creative generation if deterministic attempt failed
                        gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9})

                    out = self._pipeline(prompt, **gen_kwargs)
                    text = out[0].get("summary_text", "").strip()

                    # collapse repeated fragments, then clean
                    text = self._collapse_repeated_phrases(text)
                    cleaned = self._clean_abstractive_text(text)

                    # Quick heuristic checks (repetition, too short, or domain-like web boilerplate -> retry)
                    if self._is_repetitive_text(cleaned) or len(cleaned.split()) < 20 or self._contains_domain_noise(cleaned):
                        # try again (next attempt) with sampling
                        if attempt + 1 < attempts:
                            continue

                    # Attempt to parse structured YAML/JSON
                    overview, keywords = self._parse_structured_output(cleaned, {
                        "key_points": key_points,
                        "decisions": decisions,
                        "action_items": action_items,
                    })

                    # Final normalization / optional polish
                    overview = self._normalize_overview_text(overview)
                    if getattr(self.config, "polish_overview", True):
                        overview = self._polish_overview(overview, full_text)

                    # Validate overview quality: non-empty, not too short, not repetitive
                    if overview and len(overview.split()) >= 10 and not self._is_repetitive_text(overview):
                        return overview, keywords
                    else:
                        # Try next attempt if available, otherwise break to fallback
                        if attempt + 1 < attempts:
                            continue
                        else:
                            break
        except Exception:
            pass

        # Fallback rule-based assembly: construct a narrative paragraph summarizing meeting,
        # rather than repeating the list headers. Use polishing to turn it into an executive paragraph.
        def _format_action_items(ai_list):
            pairs = []
            for a in ai_list:
                owner = a.get('owner', 'TBD')
                task = a.get('task', '').strip()
                if task:
                    pairs.append(f"{owner} akan {task.rstrip('.')}.")
            return " ".join(pairs)

        def _join_points(pts):
            # join key points into a sentence
            if not pts:
                return ""
            # take up to 4 points to avoid overly long lists
            pts_sample = pts[:4]
            return "; ".join([p.rstrip('.') for p in pts_sample]) + ""

        narrative_parts = []
        if topics:
            narrative_parts.append("Topik utama yang dibahas meliputi: " + ", ".join(topics) + ".")
        if key_points:
            narrative_parts.append("Beberapa poin penting termasuk: " + _join_points(key_points) + ".")
        if decisions:
            narrative_parts.append("Keputusan utama yang dicapai termasuk: " + ", ".join([d.rstrip('.') for d in decisions]) + ".")
        if action_items:
            narrative_parts.append("Tindak lanjut yang disepakati di antaranya: " + _format_action_items(action_items))

        assembled = " ".join([p for p in narrative_parts if p]).strip()
        # Normalize and then optionally polish into a smooth executive paragraph
        assembled = self._normalize_overview_text(assembled)
        if getattr(self.config, "polish_overview", True):
            assembled = self._polish_overview(assembled, full_text)

        keywords = self._generate_keywords(assembled, top_k=8)
        return assembled, keywords

    def summarize(self, transcript_segments: List[TranscriptSegment]) -> MeetingSummary:
        self._load_model()

        full_text = " ".join([seg.text for seg in transcript_segments if seg.text])
        if not full_text.strip():
            return MeetingSummary(
                overview="Tidak ada konten yang dapat diringkas.",
                key_points=[],
                decisions=[],
                action_items=[],
            )

        # Clean up common disfluencies/politeness tokens and ASR annotations
        full_text = re.sub(r"\[OVERLAP\]|\[NOISE\]|<.*?>", "", full_text)
        full_text = re.sub(
            r"\b(oke|ya|oke,|baik|sekarang|sekarang kita|nah|jadi|oke\.|jadi\.)\b",
            "",
            full_text,
            flags=re.IGNORECASE,
        )
        full_text = re.sub(r"\s+", " ", full_text).strip()

        # Chunk and summarize
        if self._pipeline is None:
            # fallback: return first few sentences
            sentences = BERTSummarizer(self.config)._split_sentences(full_text)
            overview = " ".join(sentences[: min(3, len(sentences))])
        else:
            chunks = self._chunk_text(full_text)
            partial_summaries = []
            for chunk in chunks:
                try:
                    out = self._pipeline(
                        chunk,
                        max_length=self.config.max_summary_length,
                        min_length=self.config.min_summary_length,
                        truncation=True,
                        do_sample=False,
                    )
                    partial_summaries.append(out[0]["summary_text"].strip())
                except Exception as e:
                    print(f"[Summarizer] chunk summarization failed: {e}")
                    continue

            # If multiple partial summaries, join and optionally summarize again
            combined = " ".join(partial_summaries)
            if len(combined) > self.config.max_input_chars and self._pipeline:
                try:
                    out = self._pipeline(
                        combined,
                        max_length=self.config.max_summary_length,
                        min_length=self.config.min_summary_length,
                        truncation=True,
                        do_sample=False,
                    )
                    overview = out[0]["summary_text"].strip()
                except Exception:
                    overview = combined
            else:
                overview = combined

        # Clean abstractive overview and produce robust key points (use helper)
        overview, key_points = self._clean_abstractive_output(overview, full_text)

        # Extract decisions and actions via keywords
        sentences = BERTSummarizer(self.config)._split_sentences(full_text)
        decisions = BERTSummarizer(self.config)._extract_decisions(sentences)
        action_items = BERTSummarizer(self.config)._extract_action_items(transcript_segments)
        topics = BERTSummarizer(self.config)._extract_topics(full_text)

        # Optionally produce a comprehensive overview (uses abstractive pipeline)
        if getattr(self.config, "comprehensive_overview", False):
            try:
                comp_overview, keywords = self.generate_comprehensive_summary(full_text, key_points, decisions, action_items, topics)
                overview = comp_overview
            except Exception:
                keywords = []

        ms = MeetingSummary(
            overview=overview,
            key_points=key_points,
            decisions=decisions,
            action_items=action_items,
            topics=topics,
        )
        if 'keywords' in locals():
            setattr(ms, 'keywords', keywords)
        return ms


class BERTSummarizer:
    """
    Extractive Summarization using BERT sentence embeddings.

    Selects most important sentences based on semantic similarity
    to document centroid and other features.

    Attributes:
        config: SummarizationConfig object

    Example:
        >>> summarizer = BERTSummarizer()
        >>> summary = summarizer.summarize(transcript_segments)
        >>> print(summary.overview)
        >>> print(summary.decisions)
    """

    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize BERTSummarizer.

        Args:
            config: SummarizationConfig object
        """
        self.config = config or SummarizationConfig()
        self._model = None

    def _load_model(self):
        """Lazy load sentence transformer model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                print(f"[Summarizer] Loading model: {self.config.sentence_model_id}")

                self._model = SentenceTransformer(self.config.sentence_model_id)

                print("[Summarizer] Model loaded successfully")

            except Exception as e:
                print(f"[Summarizer] Warning: Could not load model: {e}")
                print("[Summarizer] Using fallback mode")
                self._model = "FALLBACK"

    def _semantic_deduplicate(self, items: List[str], threshold: Optional[float] = None) -> List[str]:
        """Delegate to AbstractiveSummarizer semantic dedup for compatibility."""
        return AbstractiveSummarizer(self.config)._semantic_deduplicate(items, threshold)

    def _semantic_dedup_action_items(self, actions: List[Dict[str, str]], threshold: Optional[float] = None) -> List[Dict[str, str]]:
        """Delegate to AbstractiveSummarizer action-item dedup for compatibility."""
        return AbstractiveSummarizer(self.config)._semantic_dedup_action_items(actions, threshold)

    def _collapse_repeated_phrases(self, text: str, max_ngram: int = 6, min_repeats: int = 2) -> str:
        """Delegates to module-level collapse helper for compatibility."""
        return _collapse_repeated_phrases_global(text, max_ngram=max_ngram, min_repeats=min_repeats)

    def summarize(self, transcript_segments: List[TranscriptSegment]) -> MeetingSummary:
        """
        Generate meeting summary from transcript.

        Args:
            transcript_segments: List of transcript segments with speaker info

        Returns:
            MeetingSummary with overview, key points, decisions, and action items
        """
        # If configuration prefers abstractive summarization, delegate to AbstractiveSummarizer
        if getattr(self.config, "method", "extractive") == "abstractive":
            try:
                return AbstractiveSummarizer(self.config).summarize(transcript_segments)
            except Exception as e:
                print(
                    f"[Summarizer] Abstractive summarization failed, falling back to extractive: {e}"
                )

        self._load_model()

        # Combine all text
        full_text = " ".join([seg.text for seg in transcript_segments if seg.text])
        # Clean up disfluencies and annotations commonly appearing in ASR output
        full_text = re.sub(r"\[OVERLAP\]|\[NOISE\]|<.*?>", "", full_text)
        full_text = re.sub(r"\s+", " ", full_text).strip()

        if not full_text.strip():
            return MeetingSummary(
                overview="Tidak ada konten yang dapat diringkas.",
                key_points=[],
                decisions=[],
                action_items=[],
            )

        # Get sentence-level metadata by merging speaker turns
        sent_meta = self._get_sentences_with_meta(transcript_segments)

        if not sent_meta:
            return MeetingSummary(
                overview="Tidak ada kalimat yang dapat diidentifikasi.",
                key_points=[],
                decisions=[],
                action_items=[],
            )

        sentences = [s["text"] for s in sent_meta]

        # Compute embeddings and select a diverse set of representative sentences via MMR
        embeddings = self._compute_embeddings(sentences)
        num_select = min(max(5, self.config.num_sentences + 2), len(sentences))

        if embeddings is not None:
            selected_idx = self._mmr_selection(sentences, embeddings, k=num_select)
            key_sentences = [sentences[i] for i in selected_idx]
        else:
            # fallback: use earlier scoring
            key_sentences = self._extract_key_sentences(sentences)

        # Generate a multi-sentence overview with some ordering and cleaning
        overview = self._generate_overview(key_sentences[:3])

        # Optionally perform a light abstractive refinement on the extractive overview
        if getattr(self.config, "do_abstractive_refinement", False):
            try:
                abs_sum = AbstractiveSummarizer(self.config)
                abs_sum._load_model()
                if abs_sum._pipeline is not None and overview:
                    out = abs_sum._pipeline(
                        overview,
                        max_length=getattr(self.config, "abstractive_refine_max_len", 80),
                        min_length=30,
                        truncation=True,
                        do_sample=False,
                    )
                    # Expect a single summary text
                    if isinstance(out, list) and out:
                        raw_overview = out[0].get("summary_text", overview).strip()
                        # Use AbstractiveSummarizer's cleaning & fallback logic
                        overview_cleaned, _ = abs_sum._clean_abstractive_output(raw_overview, full_text)
                        overview = overview_cleaned
            except Exception:
                # Fail silently and use extractive overview
                pass

        # Build richer key points: include speaker attribution and short cleaned sentences
        key_points = []
        for i in selected_idx if embeddings is not None else list(range(len(key_sentences))):
            s = sentences[i]
            sp = sent_meta[i]["speaker_id"]
            # Short clean
            s_clean = re.sub(r"\s+", " ", s).strip()
            key_points.append(f"{s_clean} (oleh {sp})")

        # Extract decisions using expanded context (look for decision keywords and enumerations)
        decisions = []
        seen_decisions = set()
        for i, s in enumerate(sentences):
            s_clean = re.sub(r"\s+", " ", s).strip()
            s_lower = s_clean.lower()
            if any(kw in s_lower for kw in self.config.decision_keywords) or re.match(
                r"^(pertama|kedua|ketiga|keempat|kelima)\b", s_lower
            ):
                context = self._expand_context_for_sentence(sent_meta, i, window=1)
                dec_text = re.sub(r"\[.*?\]", "", context)
                dec_text = re.sub(r"\s+", " ", dec_text).strip()
                # Truncate to a reasonable length (35 words) and remove trailing punctuation
                words = dec_text.split()
                dec_text = " ".join(words[:35]).rstrip(" ,.;:")
                if len(dec_text.split()) < 3:
                    continue
                if dec_text and dec_text not in seen_decisions:
                    decisions.append(dec_text)
                    seen_decisions.add(dec_text)

        # If no decisions found, try to extract from key_sentences
        if not decisions:
            for ks in key_sentences:
                if any(kw in ks.lower() for kw in self.config.decision_keywords):
                    if ks not in seen_decisions:
                        decisions.append(ks)
                        seen_decisions.add(ks)

        # Apply semantic deduplication to decisions
        try:
            decisions = self._semantic_deduplicate(decisions)
        except Exception:
            pass

        # Extract action items at sentence level with speaker inference
        action_items = []
        seen_tasks = set()
        action_kw_re = re.compile(
            r"\b(" + "|".join([re.escape(k) for k in self.config.action_keywords]) + r")\b",
            flags=re.IGNORECASE,
        )

        # verbs that indicate an actionable commitment (used to validate generic keyword matches)
        action_verbs_re = re.compile(r"\b(akan|harus|siapkan|bikin|buat|selesaikan|dikerjakan|tolong|mohon|harap)\b", flags=re.IGNORECASE)

        for i, s in enumerate(sentences):
            text = re.sub(r"\[OVERLAP\]|\[NOISE\]|<.*?>", "", s).strip()
            if not text:
                continue

            # explicit commit patterns
            commit_re = re.compile(
                r"\b(aku|saya|kami|kita|kamu)\b.*\b(bertanggung jawab|akan|saya akan|aku akan|aku akan membuat|kamu tolong|tolong|siapkan|bikin|harus|selesaikan|dikerjakan)\b",
                flags=re.IGNORECASE,
            )

            owner = None
            task = None

            if commit_re.search(text):
                owner = sent_meta[i]["speaker_id"]
                # try to isolate the actionable clause
                task = re.sub(
                    r"^.*?\b(bertanggung jawab|akan|saya akan|aku akan|kamu tolong|tolong|siapkan|bikin|harus|selesaikan|dikerjakan)\b",
                    "",
                    text,
                    flags=re.IGNORECASE,
                )
                task = task.strip(" .,:;-")
                if not task:
                    task = text

            elif action_kw_re.search(text):
                # Validate generic matches for actionability using helper
                if not self._is_actionable_text(text):
                    continue
                owner = sent_meta[i]["speaker_id"]
                task = text

            if task:
                # Normalize task text
                task = re.sub(
                    r"^\s*(aku|saya|kami|kita|kamu)\b[:,\s]*", "", task, flags=re.IGNORECASE
                ).strip()
                task = re.sub(r"\s+", " ", task).strip(" .,:;-")
                if len(task.split()) < 3:
                    continue
                filler_short = {"setuju", "oke", "ya", "nah", "betul"}
                if task.lower() in filler_short:
                    continue
                key = task.lower()[:120]
                if key in seen_tasks:
                    continue
                seen_tasks.add(key)
                action_items.append(
                    {
                        "owner": owner or "TBD",
                        "task": task,
                        "timestamp": f"{sent_meta[i]['start']:.1f}s",
                        "due": "",
                    }
                )

        # Fall back to segment-level action extraction if none found
        if not action_items:
            action_items = self._extract_action_items(transcript_segments)

        # Apply semantic deduplication to action items (merge owners when possible)
        try:
            action_items = self._semantic_dedup_action_items(action_items)
        except Exception:
            pass

        # Extract topics (frequency-based) from cleaned full_text
        topics = self._extract_topics(full_text)

        # Optionally produce a comprehensive overview (may use abstractive pipeline)
        if getattr(self.config, "comprehensive_overview", False):
            try:
                abs_s = AbstractiveSummarizer(self.config)
                comp_overview, keywords = abs_s.generate_comprehensive_summary(full_text, key_points, decisions, action_items, topics)
                overview = comp_overview
            except Exception:
                keywords = []

        # Return comprehensive MeetingSummary
        ms = MeetingSummary(
            overview=overview,
            key_points=key_points,
            decisions=decisions,
            action_items=action_items,
            topics=topics,
        )
        if 'keywords' in locals():
            setattr(ms, 'keywords', keywords)
        return ms

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Indonesian sentence splitting
        # Handle common abbreviations
        text = re.sub(r"([Dd]r|[Pp]rof|[Bb]pk|[Ii]bu|[Ss]dr|[Nn]o|[Hh]al)\.", r"\1<PERIOD>", text)

        # Split on sentence-ending punctuation
        sentences = re.split(r"[.!?]+\s*", text)

        # Restore periods in abbreviations
        sentences = [s.replace("<PERIOD>", ".") for s in sentences]

        # Clean and filter
        cleaned = []
        for s in sentences:
            s = s.strip()

            # Filter by length
            if len(s) < self.config.min_sentence_length:
                continue
            if len(s) > self.config.max_sentence_length:
                # Truncate very long sentences
                s = s[: self.config.max_sentence_length] + "..."

            # Collapse trivial repeated fragments inside sentence
            s = self._collapse_repeated_phrases(s)

            cleaned.append(s)

        return cleaned

    def _merge_speaker_turns(self, segments: List[TranscriptSegment]) -> List[Dict[str, Any]]:
        """Merge consecutive segments by the same speaker into 'turns' to provide more context.

        Returns a list of dicts: {speaker_id, start, end, text, indices}
        """
        turns: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments):
            if not seg.text or not seg.text.strip():
                continue
            # Clean common ASR artifacts and leading fillers
            text = re.sub(r"\[OVERLAP\]|\[NOISE\]|<.*?>", "", seg.text)
            text = re.sub(
                r"^\s*(oke|ya|nah|oke,|baik|sekarang|jadi)\b[\s,:-]*", "", text, flags=re.IGNORECASE
            )
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                continue

            if turns and turns[-1]["speaker_id"] == seg.speaker_id:
                turns[-1]["end"] = seg.end
                turns[-1]["text"] += " " + text
                turns[-1]["indices"].append(i)
            else:
                turns.append(
                    {
                        "speaker_id": seg.speaker_id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": text,
                        "indices": [i],
                    }
                )
        return turns

    def _get_sentences_with_meta(self, segments: List[TranscriptSegment]) -> List[Dict[str, Any]]:
        """Split merged speaker turns into sentences and keep metadata."""
        turns = self._merge_speaker_turns(segments)
        sent_meta: List[Dict[str, Any]] = []
        for t in turns:
            sents = self._split_sentences(t["text"])
            for j, s in enumerate(sents):
                sent_meta.append(
                    {
                        "text": s,
                        "speaker_id": t["speaker_id"],
                        "start": t["start"],
                        "end": t["end"],
                        "turn_indices": t["indices"],
                        "sent_idx_in_turn": j,
                    }
                )
        return sent_meta

    def _compute_embeddings(self, sentences: List[str]):
        """Compute sentence embeddings using sentence-transformers (lazy load)."""
        if not sentences:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.config.sentence_model_id)
            embs = model.encode(sentences, show_progress_bar=False)
            return embs
        except Exception as e:
            print(f"[Summarizer] Embedding model error: {e}")
            return None

    def _mmr_selection(
        self, sentences: List[str], embeddings, k: int = 5, lambda_param: float = 0.6
    ) -> List[int]:
        """Maximal Marginal Relevance (MMR) selection for diversity and coverage.

        Returns list of selected sentence indices in original order.
        """
        import numpy as _np

        if embeddings is None or len(sentences) <= k:
            return list(range(min(len(sentences), k)))

        centroid = _np.mean(embeddings, axis=0)
        # similarity to centroid
        sim_to_centroid = _np.dot(embeddings, centroid) / (
            _np.linalg.norm(embeddings, axis=1) * (_np.linalg.norm(centroid) + 1e-8)
        )

        selected = []
        candidate_indices = list(range(len(sentences)))

        # pick the top similarity as first
        first = int(_np.argmax(sim_to_centroid))
        selected.append(first)
        candidate_indices.remove(first)

        while len(selected) < k and candidate_indices:
            mmr_scores = []
            for idx in candidate_indices:
                sim_to_sel = max(
                    [
                        _np.dot(embeddings[idx], embeddings[s])
                        / (_np.linalg.norm(embeddings[idx]) * _np.linalg.norm(embeddings[s]) + 1e-8)
                        for s in selected
                    ]
                )
                score = lambda_param * sim_to_centroid[idx] - (1 - lambda_param) * sim_to_sel
                mmr_scores.append((idx, score))

            idx_best, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(idx_best)
            candidate_indices.remove(idx_best)

        # return in original order
        selected_sorted = sorted(selected)
        return selected_sorted

    def _expand_context_for_sentence(
        self, sent_meta: List[Dict[str, Any]], idx: int, window: int = 1
    ) -> str:
        """Return concatenated sentence with neighboring contextual sentences for better decision/action extraction."""
        start = max(0, idx - window)
        end = min(len(sent_meta), idx + window + 1)
        return " ".join([s["text"] for s in sent_meta[start:end]])

    def _infer_owner_for_action(self, seg_index: int, sent_meta: List[Dict[str, Any]]) -> str:
        """Infer owner for an action by looking at the sentence speaker and recent explicit mentions."""
        # Prefer sentence speaker
        if 0 <= seg_index < len(sent_meta):
            return sent_meta[seg_index]["speaker_id"]
        return "TBD"

    def _extract_key_sentences(self, sentences: List[str]) -> List[str]:
        """Extract most important sentences using BERT embeddings"""
        if not sentences:
            return []

        # Fallback mode: simple heuristics
        if self._model == "FALLBACK" or len(sentences) <= self.config.num_sentences:
            return sentences[: self.config.num_sentences]

        try:
            # Get sentence embeddings
            embeddings = self._model.encode(sentences, show_progress_bar=False)

            # Calculate document centroid
            centroid = np.mean(embeddings, axis=0)

            # Calculate importance scores for each sentence
            scores = []

            for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
                score = self._calculate_sentence_score(
                    sentence=sent,
                    embedding=emb,
                    centroid=centroid,
                    position=i,
                    total_sentences=len(sentences),
                )
                scores.append((i, score, sent))

            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)

            # Get top-k sentences (maintain original order)
            top_indices = sorted([s[0] for s in scores[: self.config.num_sentences]])

            return [sentences[i] for i in top_indices]

        except Exception as e:
            print(f"[Summarizer] Embedding extraction failed: {e}")
            return sentences[: self.config.num_sentences]

    def _calculate_sentence_score(
        self,
        sentence: str,
        embedding: np.ndarray,
        centroid: np.ndarray,
        position: int,
        total_sentences: int,
    ) -> float:
        """Calculate importance score for a sentence"""

        # 1. Cosine similarity to centroid
        similarity = np.dot(embedding, centroid) / (
            np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8
        )

        # 2. Position score (favor beginning and end)
        if total_sentences > 1:
            normalized_pos = position / (total_sentences - 1)
            # U-shaped curve: high at start and end
            position_score = 1.0 - 0.6 * np.sin(np.pi * normalized_pos)
        else:
            position_score = 1.0

        # 3. Length score (favor medium-length sentences)
        word_count = len(sentence.split())
        optimal_length = 20
        length_score = 1.0 - min(abs(word_count - optimal_length) / 30, 1.0)

        # 4. Keyword bonus
        keyword_score = 0.0
        sentence_lower = sentence.lower()

        for kw in self.config.decision_keywords + self.config.action_keywords:
            if kw in sentence_lower:
                keyword_score += 0.1

        keyword_score = min(keyword_score, 0.3)  # Cap bonus

        # Combined score
        score = (
            self.config.similarity_weight * similarity
            + self.config.position_weight * position_score
            + self.config.length_weight * length_score
            + keyword_score
        )

        return score

    def _generate_overview(self, key_sentences: List[str]) -> str:
        """Generate overview from key sentences"""
        if not key_sentences:
            return "Tidak ada ringkasan yang dapat dibuat."

        # Use top 2-3 sentences for overview
        overview_sentences = key_sentences[: min(3, len(key_sentences))]
        overview = " ".join(overview_sentences)

        # Clean up
        overview = re.sub(r"\s+", " ", overview).strip()

        return overview

    def _extract_decisions(self, sentences: List[str]) -> List[str]:
        """Extract decision-related sentences and synthesize enumerated decisions.

        This method collects sentence-level decision mentions, attempts to synthesize
        clauses from enumerated statements (e.g., "Pertama..., Kedua..."),
        and performs semantic deduplication to avoid repeated/near-duplicate items.
        """
        raw = []

        for sent in sentences:
            sent_lower = sent.lower()

            # Check for decision keywords
            if any(kw in sent_lower for kw in self.config.decision_keywords):
                # Clean the sentence
                clean_sent = re.sub(r"\s+", " ", sent).strip()
                if clean_sent and clean_sent not in raw:
                    raw.append(clean_sent)

        # Try to synthesize enumerated decisions from sentences
        synthesized = self._synthesize_enumerated_decisions(sentences)

        all_decisions = raw + synthesized

        # Deduplicate semantically (Jaccard over tokens)
        deduped = self._deduplicate_strings(all_decisions)

        # Limit number of decisions returned
        return deduped[:7]

    def _synthesize_enumerated_decisions(self, sentences: List[str]) -> List[str]:
        """Extract clauses following enumerations like 'Pertama..., Kedua...' and return list.

        Handles both ordinal words (pertama, kedua, ...) and numbered lists (1., 2.)
        by splitting and returning non-trivial clauses.
        """
        synth: List[str] = []
        enum_words_re = re.compile(r"\b(pertama|kedua|ketiga|keempat|kelima)\b", flags=re.IGNORECASE)

        for s in sentences:
            s_clean = s.strip()
            if enum_words_re.search(s_clean.lower()):
                # Split by Indonesian ordinal words
                parts = re.split(r"\bpertama\b|\bkedua\b|\bketiga\b|\bkeempat\b|\bkelima\b", s_clean, flags=re.IGNORECASE)
                for p in parts:
                    p = p.strip(" .,:;\n-–—")
                    if len(p.split()) >= 3 and p not in synth:
                        synth.append(p)

            # Also handle simple numbered enumerations like '1. ... 2. ...'
            if re.search(r"\d+\.\s*", s_clean):
                parts = re.split(r"\d+\.\s*", s_clean)
                for p in parts:
                    p = p.strip(" .,:;\n-–—")
                    if len(p.split()) >= 3 and p not in synth:
                        synth.append(p)

        return synth

    def _normalize_text_for_dedup(self, text: str) -> str:
        """Normalize text for lightweight semantic deduplication."""
        t = text.lower()
        # remove punctuation, keep alphanumerics and spaces
        t = re.sub(r"[^a-z0-9\s]+", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _deduplicate_strings(self, items: List[str], threshold: float = 0.5) -> List[str]:
        """Deduplicate items using token Jaccard similarity threshold."""
        kept: List[str] = []
        norms: List[str] = []

        for it in items:
            n = self._normalize_text_for_dedup(it)
            if not n:
                continue
            toks1 = set(n.split())
            is_dup = False
            for other in norms:
                toks2 = set(other.split())
                if not toks1 or not toks2:
                    continue
                inter = len(toks1 & toks2)
                union = len(toks1 | toks2)
                if union > 0 and (inter / union) >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(it)
                norms.append(n)

        return kept

    def _extract_action_items(self, segments: List[TranscriptSegment]) -> List[Dict[str, str]]:
        """Extract action items with speaker attribution (improved heuristics)

        Heuristics:
        - Detect explicit commitments like "aku akan", "saya bertanggung jawab", "kamu siapkan" and assign owner
        - Fallback to keyword-based detection
        - Normalize duplicate tasks and detect simple due-date mentions like "minggu depan", "besok"
        - Try to infer explicit owner names mentioned in the clause
        """
        action_items: List[Dict[str, str]] = []
        seen_tasks = set()

        # Try to use AdvancedNLPExtractor (NER + dependency parse) for higher-quality extraction
        try:
            from src.nlp_utils import AdvancedNLPExtractor

            extractor = AdvancedNLPExtractor()
            sent_meta = self._get_sentences_with_meta(segments)
            nlp_actions = extractor.extract_actions_from_sentences(sent_meta)
            for item in nlp_actions:
                task_key = item.get("task", "").lower()[:120]
                if task_key in seen_tasks:
                    continue
                seen_tasks.add(task_key)
                action_items.append(
                    {
                        "owner": item.get("owner", "TBD"),
                        "task": item.get("task", "").strip(),
                        "timestamp": f"{sent_meta[item.get('sentence_idx', 0)]['start']:.1f}s",
                        "due": self._detect_due_from_text(item.get("task", "")),
                    }
                )
        except Exception:
            extractor = None

        commit_re = re.compile(
            r"\b(aku|saya|kami|kita|kamu)\b.*\b(bertanggung jawab|akan|saya akan|aku akan|aku akan membuat|kamu tolong|tolong|siapkan|bikin|harus|selesaikan|dikerjakan)\b",
            flags=re.IGNORECASE,
        )

        # Actionable verbs/phrases to validate generic keyword matches
        _action_verbs_re = re.compile(r"\b(akan|harus|siapkan|bikin|buat|selesaikan|dikerjakan|tolong|mohon|harap)\b", flags=re.IGNORECASE)

        for seg in segments:
            if not seg.text:
                continue

            text = re.sub(r"\[OVERLAP\]|\[NOISE\]|<.*?>", "", seg.text).strip()
            text_lower = text.lower()

            # 1) explicit commitment patterns
            if commit_re.search(text_lower):
                # Try to extract short actionable clause
                task = re.sub(
                    r"^.*?(bertanggung jawab|akan|membuat|siapkan|tolong|saya akan|aku akan|kamu tolong)\b",
                    "",
                    text,
                    flags=re.IGNORECASE,
                )
                task = task.strip(" .,:;-")
                if not task:
                    # fallback to whole segment
                    task = text

                # Try to detect explicit owner name within the clause (e.g., "Budi akan ...")
                owner = self._extract_name_as_owner(text) or seg.speaker_id

                task_key = task.lower()[:120]
                if task_key not in seen_tasks:
                    seen_tasks.add(task_key)
                    action_items.append(
                        {
                            "owner": owner,
                            "task": task,
                            "timestamp": f"{seg.start:.1f}s",
                            "due": self._detect_due_from_text(task),
                        }
                    )
                continue

            # 2) keyword-based detection
            if any(kw in text_lower for kw in self.config.action_keywords):
                # Validate that the segment is actionable (has verbs like 'akan'/'perlu' or explicit name)
                if not self._is_actionable_text(text):
                    continue

                owner = self._extract_name_as_owner(text) or seg.speaker_id
                task = text.strip()
                task_key = task.lower()[:120]
                if task_key in seen_tasks:
                    continue
                seen_tasks.add(task_key)
                action_items.append(
                    {
                        "owner": owner,
                        "task": task,
                        "timestamp": f"{seg.start:.1f}s",
                        "due": self._detect_due_from_text(task),
                    }
                )

        # Post-process: deduplicate semantically and filter tiny filler tasks
        processed: List[Dict[str, str]] = []
        seen_norms = set()

        # Filter out filler / non-actionable phrases (e.g., meeting start/thanks)
        filler_patterns = [
            r"\bkita mulai rapat",
            r"\bitu yang mau kita bahas",
            r"\bterima kasih",
            r"\bok(e|ey)?\b",
            r"\bsip\b",
            r"\bcukup(kan)? sampai",
            r"\btidak ada( yang)?\b",
            r"\biya\b",
            r"\bsetuju\b",
        ]
        filler_re = re.compile("|".join(filler_patterns), flags=re.IGNORECASE)

        for it in action_items:
            task_text = it.get("task", "")

            # Skip common non-actionable conversational lines
            if filler_re.search(task_text):
                continue

            # Ensure the sentence is actionable (has a commitment verb or explicit owner/name)
            if not self._is_actionable_text(task_text):
                continue

            norm = self._normalize_text_for_dedup(task_text)[:200]
            # skip if too short
            if len(task_text.split()) < 3:
                continue
            if norm in seen_norms:
                continue
            seen_norms.add(norm)
            processed.append(it)

        # Limit number of action items
        return processed[:15]

    def _detect_due_from_text(self, text: str) -> str:
        """Detect simple due-date hints from text and return a short normalized due string."""
        t = text.lower()
        if "besok" in t:
            return "besok"
        if "segera" in t or "secepat" in t or "sekarang" in t:
            return "segera"
        if "minggu depan" in t:
            return "1 minggu"
        m = re.search(r"(\d+)\s*minggu", t)
        if m:
            return f"{m.group(1)} minggu"
        if "2 minggu" in t or "dua minggu" in t:
            return "2 minggu"
        if "deadline" in t:
            # try to capture a following date/token
            m2 = re.search(r"deadline\s*[:\-\s]*([\w\-\./]+)", t)
            return m2.group(1) if m2 else "TBD"
        return ""

    def _extract_name_as_owner(self, text: str) -> Optional[str]:
        """Return a candidate owner name if a capitalized proper name is explicitly present in the clause.

        Simple heuristic: look for capitalized words (not at sentence start if it's a pronoun) followed by 'akan' or similar.
        """
        m = re.search(r"\b([A-Z][a-z]{2,})\b(?=\s+akan|\s+siapkan|\s+tolong|\s+bisa|\s+bertanggung)", text)
        if m:
            return m.group(1)
        return None

    def _is_actionable_text(self, text: str) -> bool:
        """Return True if text contains indicators of an actionable commitment.

        Indicators:
        - Commitment verbs (akan, harus, perlu, siapkan, dll.)
        - Explicit owner mention (capitalized name)
        - Time indicators / deadlines (besok, minggu depan, deadline)
        """
        t = text or ""
        tl = t.lower()
        if re.search(r"\b(akan|harus|siapkan|bikin|buat|selesaikan|dikerjakan|tolong|mohon|harap|perlu)\b", tl):
            return True
        # Only consider capitalized names as indicators if followed by an action verb
        if re.search(r"\b([A-Z][a-z]{2,})\b(?=\s+(akan|siapkan|tolong|mohon|harus|selesaikan|buat|bikin))", t):
            return True
        if any(k in tl for k in ("deadline", "minggu depan", "besok")):
            return True
        return False

    def _extract_topics(self, text: str, num_topics: int = 5) -> List[str]:
        """Extract main topics from text using simple frequency analysis"""
        # Simple word frequency approach
        # Remove common Indonesian stopwords
        stopwords = {
            "yang",
            "dan",
            "di",
            "ke",
            "dari",
            "ini",
            "itu",
            "dengan",
            "untuk",
            "pada",
            "adalah",
            "dalam",
            "tidak",
            "akan",
            "sudah",
            "juga",
            "saya",
            "kita",
            "kami",
            "mereka",
            "ada",
            "bisa",
            "atau",
            "seperti",
            "jadi",
            "kalau",
            "karena",
            "tapi",
            "ya",
            "apa",
            "bagaimana",
            "kenapa",
            "siapa",
            "kapan",
            "dimana",
            "nya",
            "kan",
            "dong",
            "sih",
            "kok",
            "deh",
            "loh",
            "lah",
        }

        # Tokenize and count
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        word_counts = {}

        for word in words:
            if word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Return top topics
        return [word for word, count in sorted_words[:num_topics]]

    def summarize_by_speaker(self, segments: List[TranscriptSegment]) -> Dict[str, str]:
        """Generate per-speaker summary"""
        # Group segments by speaker
        speaker_texts = {}

        for seg in segments:
            if seg.speaker_id not in speaker_texts:
                speaker_texts[seg.speaker_id] = []
            speaker_texts[seg.speaker_id].append(seg.text)

        # Summarize each speaker's contribution
        speaker_summaries = {}

        for speaker_id, texts in speaker_texts.items():
            full_text = " ".join(texts)
            sentences = self._split_sentences(full_text)

            if sentences:
                # Get top 2 sentences for each speaker
                key_sentences = self._extract_key_sentences(sentences)[:2]
                speaker_summaries[speaker_id] = " ".join(key_sentences)
            else:
                speaker_summaries[speaker_id] = "Tidak ada kontribusi yang dapat diringkas."

        return speaker_summaries
