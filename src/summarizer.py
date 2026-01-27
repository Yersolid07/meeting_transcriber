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

from src.transcriber import TranscriptSegment


@dataclass
class SummarizationConfig:
    """Configuration for summarization"""

    # Method: 'extractive' (BERT embeddings) or 'abstractive' (seq2seq model)
    method: str = "abstractive"

    # Models
    # Use a cached/available model for reliability in offline environments
    sentence_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    abstractive_model_id: str = "google/mt5-small"

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
                lines.append(f"  {i}. [{owner}] {task}")

        return "\n".join(lines)


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
                chunks.append(chunk)
            start = cut
        return chunks

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

        # Extract sentences and key points heuristically from overview
        key_points = [s.strip() for s in re.split(r"\.|!|\?", overview) if s.strip()][
            : self.config.num_sentences
        ]

        # Extract decisions and actions via keywords
        sentences = BERTSummarizer(self.config)._split_sentences(full_text)
        decisions = BERTSummarizer(self.config)._extract_decisions(sentences)
        action_items = BERTSummarizer(self.config)._extract_action_items(transcript_segments)
        topics = BERTSummarizer(self.config)._extract_topics(full_text)

        return MeetingSummary(
            overview=overview,
            key_points=key_points,
            decisions=decisions,
            action_items=action_items,
            topics=topics,
        )


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
                        overview = out[0].get("summary_text", overview).strip()
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

        # Extract action items at sentence level with speaker inference
        action_items = []
        seen_tasks = set()
        action_kw_re = re.compile(
            r"\b(" + "|".join([re.escape(k) for k in self.config.action_keywords]) + r")\b",
            flags=re.IGNORECASE,
        )

        for i, s in enumerate(sentences):
            text = re.sub(r"\[OVERLAP\]|\[NOISE\]|<.*?>", "", s).strip()
            if not text:
                continue

            # explicit commit patterns
            commit_re = re.compile(
                r"\b(aku|saya|kami|kita|kamu)\b.*\b(bertanggung jawab|akan|saya akan|aku akan|aku akan membuat|kamu tolong|tolong|siapkan|bikin)\b",
                flags=re.IGNORECASE,
            )

            owner = None
            task = None

            if commit_re.search(text):
                owner = sent_meta[i]["speaker_id"]
                # try to isolate the actionable clause
                task = re.sub(
                    r"^.*?\b(bertanggung jawab|akan|saya akan|aku akan|kamu tolong|tolong|siapkan|bikin)\b",
                    "",
                    text,
                    flags=re.IGNORECASE,
                )
                task = task.strip(" .,:;-")
                if not task:
                    task = text

            elif action_kw_re.search(text):
                # Use sentence as task, infer owner as the speaker of this sentence
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

        # Extract topics (frequency-based) from cleaned full_text
        topics = self._extract_topics(full_text)

        # Return comprehensive MeetingSummary
        return MeetingSummary(
            overview=overview,
            key_points=key_points,
            decisions=decisions,
            action_items=action_items,
            topics=topics,
        )

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
        """Extract decision-related sentences"""
        decisions = []

        for sent in sentences:
            sent_lower = sent.lower()

            # Check for decision keywords
            if any(kw in sent_lower for kw in self.config.decision_keywords):
                # Clean the sentence
                clean_sent = sent.strip()
                if clean_sent and clean_sent not in decisions:
                    decisions.append(clean_sent)

        # Limit number of decisions
        return decisions[:7]

    def _extract_action_items(self, segments: List[TranscriptSegment]) -> List[Dict[str, str]]:
        """Extract action items with speaker attribution (improved heuristics)

        Heuristics:
        - Detect explicit commitments like "aku akan", "saya bertanggung jawab", "kamu siapkan" and assign owner
        - Fallback to keyword-based detection
        - Remove overlapping/annotation tokens and short filler phrases
        """
        action_items = []
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
                        "due": "",
                    }
                )
        except Exception:
            extractor = None

        commit_re = re.compile(
            r"\b(aku|saya|kami|kita|kamu)\b.*\b(bertanggung jawab|akan|saya akan|aku akan|aku akan membuat|kamu tolong|tolong|siapkan|bikin)\b",
            flags=re.IGNORECASE,
        )

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

                task_key = task.lower()[:80]
                if task_key not in seen_tasks:
                    seen_tasks.add(task_key)
                    action_items.append(
                        {
                            "owner": seg.speaker_id,
                            "task": task,
                            "timestamp": f"{seg.start:.1f}s",
                            "due": "",
                        }
                    )
                continue

            # 2) keyword-based detection
            if any(kw in text_lower for kw in self.config.action_keywords):
                task = text.strip()
                task_key = task.lower()[:80]
                if task_key in seen_tasks:
                    continue
                seen_tasks.add(task_key)
                action_items.append(
                    {
                        "owner": seg.speaker_id,
                        "task": task,
                        "timestamp": f"{seg.start:.1f}s",
                        "due": "",
                    }
                )

        # Limit number of action items
        return action_items[:15]

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
