#!/usr/bin/env python3
"""
Unit tests for Meeting Transcriber Pipeline
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_processor import AudioProcessor
from src.diarization import DiarizationConfig, SpeakerDiarizer, SpeakerSegment
from src.document_generator import DocumentGenerator, MeetingMetadata
from src.evaluator import Evaluator
from src.pipeline import MeetingTranscriberPipeline, PipelineConfig
from src.summarizer import BERTSummarizer, MeetingSummary
from src.transcriber import TranscriptSegment


class TestAudioProcessor(unittest.TestCase):
    """Test AudioProcessor class"""

    def setUp(self):
        self.processor = AudioProcessor()
        self.sample_rate = 16000
        self.duration = 5.0  # 5 seconds

        # Create dummy waveform
        t = torch.linspace(0, self.duration, int(self.duration * self.sample_rate))
        self.dummy_waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)

    def test_get_duration(self):
        """Test duration calculation"""
        duration = self.processor.get_duration(self.dummy_waveform, self.sample_rate)
        self.assertAlmostEqual(duration, self.duration, places=2)

    def test_cut_segment(self):
        """Test segment cutting"""
        segment = self.processor.cut_segment(self.dummy_waveform, 1.0, 3.0, self.sample_rate)

        expected_length = int(2.0 * self.sample_rate)
        self.assertEqual(segment.shape[1], expected_length)

    def test_split_into_chunks(self):
        """Test chunk splitting"""
        chunks = self.processor.split_into_chunks(
            self.dummy_waveform, chunk_duration=2.0, overlap=0.5, sample_rate=self.sample_rate
        )

        self.assertGreater(len(chunks), 0)
        for chunk, start, end in chunks:
            self.assertLessEqual(end - start, 2.0)

    def test_normalize(self):
        """Test normalization"""
        # Create waveform with values > 1
        waveform = torch.randn(1, 16000) * 2
        normalized = self.processor._normalize(waveform)

        self.assertLessEqual(torch.max(torch.abs(normalized)).item(), 1.0)


class TestSpeakerDiarizer(unittest.TestCase):
    """Test SpeakerDiarizer class"""

    def setUp(self):
        self.diarizer = SpeakerDiarizer(
            config=DiarizationConfig(device="cpu"), models_dir="./models"
        )
        self.sample_rate = 16000

        # Create dummy waveform (10 seconds)
        duration = 10.0
        t = torch.linspace(0, duration, int(duration * self.sample_rate))
        self.dummy_waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)

    def test_detect_speech(self):
        """Test VAD"""
        regions = self.diarizer._detect_speech(self.dummy_waveform, self.sample_rate)

        # Should detect some speech
        self.assertIsInstance(regions, list)

    def test_create_windows(self):
        """Test window creation"""
        speech_regions = [(0.0, 5.0), (6.0, 10.0)]
        windows = self.diarizer._create_windows(speech_regions)

        self.assertGreater(len(windows), 0)
        for start, end in windows:
            self.assertLess(start, end)

    def test_merge_segments(self):
        """Test segment merging"""
        segments = [
            SpeakerSegment("SPEAKER_00", 0.0, 2.0),
            SpeakerSegment("SPEAKER_00", 2.1, 4.0),  # Should merge
            SpeakerSegment("SPEAKER_01", 4.5, 6.0),  # Different speaker
        ]

        merged = self.diarizer._merge_segments(segments, max_gap=0.5)

        # First two should be merged
        self.assertLessEqual(len(merged), 2)


class TestEvaluator(unittest.TestCase):
    """Test Evaluator class"""

    def setUp(self):
        self.evaluator = Evaluator()

    def test_preprocess_text(self):
        """Test text preprocessing"""
        text = "Hello, World! How are you?"
        processed = self.evaluator.preprocess_text(text)

        self.assertEqual(processed, "hello world how are you")

    def test_wer_perfect(self):
        """Test WER with perfect transcription"""
        reference = "hello world"
        hypothesis = "hello world"

        result = self.evaluator.calculate_wer(reference, hypothesis)

        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.substitutions, 0)
        self.assertEqual(result.deletions, 0)
        self.assertEqual(result.insertions, 0)

    def test_wer_with_errors(self):
        """Test WER with errors"""
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on mat"  # Missing "the"

        result = self.evaluator.calculate_wer(reference, hypothesis)

        self.assertGreater(result.wer, 0.0)
        self.assertEqual(result.deletions, 1)

    def test_wer_substitution(self):
        """Test WER with substitution"""
        reference = "hello world"
        hypothesis = "hello word"  # 'world' -> 'word'

        result = self.evaluator.calculate_wer(reference, hypothesis)

        self.assertGreater(result.wer, 0.0)
        self.assertEqual(result.substitutions, 1)

    def test_der_calculation(self):
        """Test DER calculation"""
        reference = [
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ]

        hypothesis = [
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
        ]

        result = self.evaluator.calculate_der(reference, hypothesis)

        # Should be close to 0 for matching segments
        self.assertLess(result.der, 0.5)


class TestBERTSummarizer(unittest.TestCase):
    """Test BERTSummarizer class"""

    def setUp(self):
        self.summarizer = BERTSummarizer()

    def test_split_sentences(self):
        """Test sentence splitting"""
        text = "Ini kalimat pertama. Ini kalimat kedua! Ini kalimat ketiga?"
        sentences = self.summarizer._split_sentences(text)

        self.assertEqual(len(sentences), 3)

    def test_extract_decisions(self):
        """Test decision extraction"""
        sentences = [
            "Kita akan membahas proyek baru.",
            "Diputuskan bahwa proyek akan dimulai minggu depan.",
            "Tim setuju dengan timeline ini.",
        ]

        decisions = self.summarizer._extract_decisions(sentences)

        self.assertGreater(len(decisions), 0)

    def test_empty_input(self):
        """Test with empty input"""
        segments = []
        summary = self.summarizer.summarize(segments)

        self.assertIsInstance(summary, MeetingSummary)
        self.assertIn("tidak ada", summary.overview.lower())


class TestDocumentGenerator(unittest.TestCase):
    """Test DocumentGenerator class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = DocumentGenerator(output_dir=self.temp_dir)

    def tearDown(self):
        # Cleanup temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_format_timestamp(self):
        """Test timestamp formatting"""
        timestamp = self.generator._format_timestamp(65.5, 125.3)

        self.assertIn("01:05", timestamp)
        self.assertIn("02:05", timestamp)

    def test_sanitize_filename(self):
        """Test filename sanitization"""
        filename = "Meeting: Test/File?Name"
        sanitized = self.generator._sanitize_filename(filename)

        self.assertNotIn(":", sanitized)
        self.assertNotIn("/", sanitized)
        self.assertNotIn("?", sanitized)

    def test_generate_document(self):
        """Test document generation"""
        metadata = MeetingMetadata(title="Test Meeting", date="2024-01-01", location="Test Room")

        summary = MeetingSummary(
            overview="Test overview",
            key_points=["Point 1", "Point 2"],
            decisions=["Decision 1"],
            action_items=[{"owner": "SPEAKER_00", "task": "Test task"}],
        )

        transcript = [
            TranscriptSegment(speaker_id="SPEAKER_00", start=0.0, end=5.0, text="Hello everyone.")
        ]

        doc_path = self.generator.generate(
            metadata=metadata,
            summary=summary,
            transcript=transcript,
            output_filename="test_output.docx",
        )

        self.assertTrue(os.path.exists(doc_path))
        self.assertTrue(doc_path.endswith(".docx"))


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for full pipeline"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = PipelineConfig(
            output_dir=self.temp_dir,
            cache_dir=self.temp_dir,
            models_dir=self.temp_dir,
            device="cpu",
            verbose=False,
            save_intermediate=False,
        )
        self.pipeline = MeetingTranscriberPipeline(self.config)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.config.device, "cpu")

    def test_clear_state(self):
        """Test state clearing"""
        # Set some state
        self.pipeline._waveform = torch.randn(1, 16000)
        self.pipeline._transcript_segments = []

        # Clear
        self.pipeline.clear_state()

        # Verify cleared
        self.assertIsNone(self.pipeline._waveform)
        self.assertIsNone(self.pipeline._transcript_segments)

    def test_get_transcript_text_empty(self):
        """Test empty transcript"""
        text = self.pipeline.get_transcript_text()
        self.assertEqual(text, "")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAudioProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestSpeakerDiarizer))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestBERTSummarizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    run_tests()
