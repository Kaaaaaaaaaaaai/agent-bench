from agent_bench.models import Task
from agent_bench.prompts import REFERENCE_PLACEHOLDER, build_user_prompt


def test_text_recall_prompt_replaces_reference_placeholder():
    task = Task(
        id="CR_001",
        category="code_recall",
        type="text_recall",
        question=(
            "Use this source code as the context window:\n"
            "<source>\n"
            f"{REFERENCE_PLACEHOLDER}\n"
            "</source>\n"
            "Reproduce the requested function."
        ),
        source="code_recall.json",
        expected_text="def target(value):",
        reference_path="ref/example.py",
        reference_text="def target(value):\n    return value\n",
    )

    prompt = build_user_prompt(task)

    assert REFERENCE_PLACEHOLDER not in prompt
    assert "Use this source code as the context window:" in prompt
    assert "ref/example.py" not in prompt
    assert "def target(value):\n    return value" in prompt
