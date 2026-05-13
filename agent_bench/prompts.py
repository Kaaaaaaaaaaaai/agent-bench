import json

from agent_bench.models import Task

REFERENCE_PLACEHOLDER = "{{REFERENCE_CODE}}"


SYSTEM_PROMPT = (
    "You are answering benchmark questions. Return only valid JSON matching the requested schema. "
    "Do not include Markdown fences, prose, or extra keys."
)


def build_messages(task: Task) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task)},
    ]


def build_user_prompt(task: Task) -> str:
    if task.is_multiple_choice:
        return _multiple_choice_prompt(task)
    if task.is_coding:
        return _coding_prompt(task)
    if task.is_text_recall:
        return _text_recall_prompt(task)
    raise ValueError(f"Unsupported task type: {task.type}")


def _multiple_choice_prompt(task: Task) -> str:
    choices = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(task.choices))
    return (
        f"Task ID: {task.id}\n"
        f"Category: {task.category}\n\n"
        f"Question:\n{task.question}\n\n"
        f"Choices:\n{choices}\n\n"
        "Respond with JSON exactly in this shape:\n"
        '{"answer": ["A"], "confidence": 0.0}\n'
        "Use one or more uppercase choice letters in answer."
    )


def _coding_prompt(task: Task) -> str:
    examples = json.dumps([case["input"] for case in task.test_cases[:3]], ensure_ascii=False, indent=2)
    class_hint = ""
    if _is_class_task(task):
        class_hint = (
            "\nThis is a class-design task. Implement a Python class named exactly "
            f"`{task.function_name}` with the methods described by the problem. The evaluator will "
            "instantiate the class and call methods using the operations and arguments test format."
        )
    else:
        class_hint = (
            "\nImplement a Python function named exactly "
            f"`{task.function_name}`. It must accept keyword arguments matching the JSON-compatible "
            "test inputs and return a JSON-compatible result. If a common LeetCode problem normally "
            "uses linked lists, arrays, or in-place mutation, use the JSON-compatible inputs shown here."
        )

    return (
        f"Task ID: {task.id}\n"
        f"Category: {task.category}\n"
        f"Title: {task.title or task.function_name}\n\n"
        f"Problem:\n{task.question}\n"
        f"{class_hint}\n\n"
        f"Example evaluator inputs:\n{examples}\n\n"
        "Respond with JSON exactly in this shape:\n"
        '{"code": "def functionName(...):\\n    ...", "confidence": 0.0}\n'
        "The code should be plain Python 3.12-compatible source."
    )


def _text_recall_prompt(task: Task) -> str:
    reference_text = task.reference_text or ""
    question = task.question.replace(REFERENCE_PLACEHOLDER, reference_text)
    return (
        f"Task ID: {task.id}\n"
        f"Category: {task.category}\n\n"
        f"Question:\n{question}\n\n"
        "Respond with JSON exactly in this shape:\n"
        '{"answer": "verbatim text here", "confidence": 0.0}\n'
        "Preserve code text, indentation, punctuation, and line breaks exactly. "
        "Do not add explanation."
    )


def _is_class_task(task: Task) -> bool:
    if not task.test_cases:
        return False
    first_input = task.test_cases[0].get("input", {})
    operations = first_input.get("operations")
    return isinstance(operations, list) and bool(operations) and operations[0] == task.function_name
