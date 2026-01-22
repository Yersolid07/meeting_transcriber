import os
import sys
import tempfile
import unittest

# Make "src" importable during tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_generator import DocumentGenerator, MeetingMetadata
from src.evaluator import DERResult, Evaluator, WERResult
from src.summarizer import MeetingSummary
from src.transcriber import ASRTranscriber


class TestAdditionalFeatures(unittest.TestCase):
    def test_docx_generation(self):
        outdir = tempfile.mkdtemp()
        gen = DocumentGenerator(output_dir=outdir)
        metadata = MeetingMetadata.create_default(audio_duration_sec=0)
        summary = MeetingSummary(
            overview="Test overview",
            key_points=[],
            decisions=[],
            action_items=[],
        )
        transcript = []

        path = gen.generate(metadata, summary, transcript, output_filename="test_doc.docx")
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_transcriber_postprocess(self):
        tr = ASRTranscriber()
        cleaned = tr._postprocess_text("Ini contoh $$math$$ teks.")
        self.assertNotIn("$$", cleaned)
        self.assertNotIn("math", cleaned)

    def test_evaluator_report(self):
        evaluator = Evaluator()

        # Build simple WER/DER lists
        wer = WERResult(wer=0.1234, substitutions=1, deletions=0, insertions=0, hits=10)
        der = DERResult(der=0.0456, missed_speech=0.01, false_alarm=0.02, speaker_confusion=0.015)

        report = evaluator.generate_evaluation_report(
            [wer], [der], sample_names=["sample1"], condition_name="cond"
        )
        self.assertIn("Kondisi", report)
        self.assertIn("cond", report)


if __name__ == "__main__":
    unittest.main()
