from src.evaluator import Evaluator


def test_calculate_summary_metrics_smoke():
    e = Evaluator(output_dir="./data/output")
    res = e.calculate_summary_metrics("Ini ringkasan referensi.", "Ringkasan sistem.")
    assert res is not None
    assert hasattr(res, "rouge")
    assert hasattr(res, "bertscore")
