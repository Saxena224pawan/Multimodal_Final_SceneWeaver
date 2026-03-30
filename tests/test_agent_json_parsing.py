from __future__ import annotations

from multi_agent_refinement.agent_base import extract_first_json_object


def test_extract_first_json_object_handles_trailing_text() -> None:
    payload = 'Preface {"beat_adherence": 82, "issues": []} trailing explanation {"ignored": true}'

    parsed = extract_first_json_object(payload)

    assert parsed == {"beat_adherence": 82, "issues": []}
