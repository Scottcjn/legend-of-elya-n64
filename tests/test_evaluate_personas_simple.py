import importlib.util
from datetime import datetime as real_datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FixedDateTime:
    @classmethod
    def now(cls):
        return real_datetime(2026, 5, 12, 1, 2)


def load_evaluator(module_name="evaluate_personas_simple_test"):
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "scripts" / "evaluate_personas_simple.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_evaluator_does_not_generate_report(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    module = load_evaluator()

    assert len(module.EVAL_PROMPTS) == 20
    assert set(module.PERSONA_RESPONSES) == {
        "sophia",
        "blacksmith",
        "librarian",
        "guard",
    }
    assert not (tmp_path / "eval_results").exists()


def test_generate_evaluation_report_writes_expected_summary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module = load_evaluator("evaluate_personas_simple_report_test")
    monkeypatch.setattr(module, "datetime", FixedDateTime)

    output_path = module.generate_evaluation_report()

    report_path = tmp_path / output_path
    report = report_path.read_text(encoding="utf-8")

    assert output_path == "eval_results/persona_eval_20260512_0102.md"
    assert report_path.exists()
    assert "**Prompts:** 20" in report
    assert "| Sophia | warm and knowledgeable | 20 |" in report
    assert "### 6. What can you tell me about RustChain?" in report
    assert "| **Guard** | RustChain? Not in security briefing. Classify as low priority. |" in report


def test_generate_evaluation_report_escapes_table_pipes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module = load_evaluator("evaluate_personas_simple_pipe_test")
    monkeypatch.setattr(module, "datetime", FixedDateTime)
    monkeypatch.setattr(module, "EVAL_PROMPTS", ["Question with a pipe?"])
    monkeypatch.setattr(
        module,
        "PERSONA_RESPONSES",
        {
            "tester": {
                "style": "brief",
                "responses": ["left | right"],
            },
        },
    )

    output_path = module.generate_evaluation_report()
    report = (tmp_path / output_path).read_text(encoding="utf-8")

    assert "| **Tester** | left \\| right |" in report
