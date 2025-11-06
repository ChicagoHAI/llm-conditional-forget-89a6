import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "conditional_forgetting.jsonl"


def square_to_coord(square: str) -> Tuple[int, int]:
    file = ord(square[0].lower()) - ord("a")
    rank = int(square[1]) - 1
    return file, rank


def knight_distance(a: str, b: str) -> Tuple[int, int]:
    ax, ay = square_to_coord(a)
    bx, by = square_to_coord(b)
    return abs(ax - bx), abs(ay - by)


def chess_scenarios() -> List[Dict]:
    scenarios: List[Dict] = []

    def add_case(rule_id: str, description: str, start: str, end: str, legal: bool) -> None:
        scenarios.append(
            {
                "id": rule_id,
                "domain": "chess",
                "rule": description,
                "question": f"A piece starts on {start.upper()} and targets {end.upper()}. Does the move succeed under this variant?",
                "choices": {
                    "A": "The move is legal under the variant rules.",
                    "B": "The move is illegal under the variant rules.",
                },
                "correct_choice": "A" if legal else "B",
                "answer_type": "multiple_choice",
                "metadata": {"start": start, "end": end},
            }
        )

    # Knight moves exactly like a bishop (diagonal any distance)
    knight_as_bishop = [
        ("c3", "f6"),
        ("b1", "c3"),
        ("e4", "b7"),
        ("g1", "e2"),
        ("d5", "a8"),
    ]
    for idx, (start, end) in enumerate(knight_as_bishop, 1):
        dx, dy = knight_distance(start, end)
        legal = dx == dy and dx > 0
        add_case(f"chess_knight_as_bishop_{idx}", "Knights move exactly like bishops (any-length diagonals).", start, end, legal)

    # Bishop moves like a knight (L-shape)
    bishop_as_knight = [
        ("c1", "d3"),
        ("f1", "c4"),
        ("a3", "b5"),
        ("e3", "f5"),
        ("h4", "g6"),
    ]
    for idx, (start, end) in enumerate(bishop_as_knight, 1):
        dx, dy = knight_distance(start, end)
        legal = sorted((dx, dy)) == [1, 2]
        add_case(f"chess_bishop_as_knight_{idx}", "Bishops move like knights (standard L-shapes).", start, end, legal)

    # Rooks limited to king moves (one square in any direction)
    rook_as_king = [
        ("a1", "a2"),
        ("d4", "e5"),
        ("h8", "g8"),
        ("c6", "d7"),
        ("e2", "e4"),
    ]
    for idx, (start, end) in enumerate(rook_as_king, 1):
        ax, ay = square_to_coord(start)
        bx, by = square_to_coord(end)
        legal = max(abs(ax - bx), abs(ay - by)) == 1
        add_case(f"chess_rook_as_king_{idx}", "Rooks may only move one square in any direction (like kings).", start, end, legal)

    # Pawns move one square backward instead of forward (ignore captures/en passant)
    pawn_backward = [
        ("white", "d4", "d3"),
        ("white", "c2", "c1"),
        ("black", "f5", "f6"),
        ("black", "b7", "b8"),
        ("white", "g5", "g6"),
    ]
    for idx, (color, start, end) in enumerate(pawn_backward, 1):
        _, sy = square_to_coord(start)
        _, ey = square_to_coord(end)
        delta = ey - sy
        if color == "white":
            legal = delta == -1
        else:
            legal = delta == 1
        add_case(
            f"chess_pawn_backward_{idx}",
            "Pawns move exactly one square backward (opposite of normal direction) and do not move forward.",
            start,
            end,
            legal,
        )

    # Queens move only like knights
    queen_as_knight = [
        ("d1", "f2"),
        ("h4", "g6"),
        ("c3", "d5"),
        ("e5", "f7"),
        ("a8", "b6"),
    ]
    for idx, (start, end) in enumerate(queen_as_knight, 1):
        dx, dy = knight_distance(start, end)
        legal = sorted((dx, dy)) == [1, 2]
        add_case(f"chess_queen_as_knight_{idx}", "Queens move exactly like knights (L-shapes only).", start, end, legal)

    return scenarios


def multiple_choice_from_values(values: List[int], correct: int) -> Tuple[Dict[str, str], str]:
    unique_values = []
    for val in values:
        if val not in unique_values:
            unique_values.append(val)
    if correct not in unique_values:
        unique_values.insert(0, correct)

    labels = ["A", "B", "C", "D"]
    options = {}
    correct_label = None
    for label, val in zip(labels, unique_values[:4]):
        options[label] = str(val)
        if val == correct:
            correct_label = label
    if correct_label is None:
        correct_label = labels[len(unique_values) % len(labels)]
        options[correct_label] = str(correct)
    return options, correct_label


def arithmetic_scenarios() -> List[Dict]:
    scenarios: List[Dict] = []

    # Addition offset: a ⊕ b = a + b - 3
    for idx, (a, b) in enumerate([(7, 6), (12, 9), (4, 15), (20, 13), (9, 2)], 1):
        correct = a + b - 3
        choices, answer = multiple_choice_from_values([correct, correct + 3, correct - 3, correct + 5], correct)
        scenarios.append(
            {
                "id": f"math_offset_add_{idx}",
                "domain": "math",
                "rule": "Redefine addition: x ⊕ y = x + y - 3.",
                "question": f"What is {a} ⊕ {b}?",
                "choices": choices,
                "correct_choice": answer,
                "answer_type": "multiple_choice",
                "metadata": {"operands": [a, b]},
            }
        )

    # Multiplication boost: a ⊗ b = (a * b) + 5
    for idx, (a, b) in enumerate([(3, 4), (6, 5), (8, 2), (7, 7), (9, 3)], 1):
        correct = a * b + 5
        choices, answer = multiple_choice_from_values([correct, a * b, correct + 10, correct - 4], correct)
        scenarios.append(
            {
                "id": f"math_boost_mult_{idx}",
                "domain": "math",
                "rule": "Redefine multiplication: x ⊗ y = (x × y) + 5.",
                "question": f"What is {a} ⊗ {b}?",
                "choices": choices,
                "correct_choice": answer,
                "answer_type": "multiple_choice",
                "metadata": {"operands": [a, b]},
            }
        )

    # Square shift: sq(n) = n^2 - 10
    for idx, n in enumerate([5, 9, 12, 4, 15], 1):
        correct = n * n - 10
        choices, answer = multiple_choice_from_values([correct, n * n, correct + 10, correct - 6], correct)
        scenarios.append(
            {
                "id": f"math_shift_square_{idx}",
                "domain": "math",
                "rule": "Redefine squaring: sq(n) = n^2 - 10.",
                "question": f"What is sq({n})?",
                "choices": choices,
                "correct_choice": answer,
                "answer_type": "multiple_choice",
                "metadata": {"operand": n},
            }
        )

    # Division redefined as difference of larger minus smaller
    for idx, (a, b) in enumerate([(9, 2), (25, 7), (18, 5), (11, 3), (30, 22)], 1):
        correct = abs(a - b)
        choices, answer = multiple_choice_from_values([correct, a // b if b else 0, correct + 2, correct - 1], correct)
        scenarios.append(
            {
                "id": f"math_diff_div_{idx}",
                "domain": "math",
                "rule": "Redefine division: x ÷ y = |x - y|.",
                "question": f"What is {a} ÷ {b} under this rule?",
                "choices": choices,
                "correct_choice": answer,
                "answer_type": "multiple_choice",
                "metadata": {"operands": [a, b]},
            }
        )

    return scenarios


def protocol_scenarios() -> List[Dict]:
    scenarios: List[Dict] = []

    def add_protocol(idx: int, title: str, question: str, options: Dict[str, str], answer: str) -> None:
        scenarios.append(
            {
                "id": f"protocol_{idx}",
                "domain": "protocol",
                "rule": title,
                "question": question,
                "choices": options,
                "correct_choice": answer,
                "answer_type": "multiple_choice",
                "metadata": {},
            }
        )

    scenarios_data = [
        (
            1,
            "In this workshop, participants must greet by stating their favorite prime number instead of saying hello.",
            "You enter the room. How should you greet the group?",
            {"A": "Say “Hello everyone!”", "B": "State your favorite prime number.", "C": "Remain silent and wave."},
            "B",
        ),
        (
            2,
            "During team standups, everyone reports blockers before successes.",
            "It's your turn to speak. What do you mention first?",
            {"A": "Share a recent win.", "B": "Present blockers you're facing.", "C": "Thank the team lead."},
            "B",
        ),
        (
            3,
            "In this lab, gloves must be worn inside-out to show the colored lining for quick contamination checks.",
            "How do you put on gloves before starting?",
            {"A": "Wear gloves normally.", "B": "Wear gloves inside-out.", "C": "Skip gloves for short tasks."},
            "B",
        ),
        (
            4,
            "Company visitors receive tea unless they explicitly request coffee.",
            "A visitor arrives and says nothing about drinks. What should you serve?",
            {"A": "Serve tea.", "B": "Serve coffee.", "C": "Ask again before serving anything."},
            "A",
        ),
        (
            5,
            "In this training simulation, emergency exits are to be referred to as “blue doors.”",
            "How do you instruct trainees to leave the building?",
            {
                "A": "Tell them to use the emergency exits.",
                "B": "Tell them to head for the blue doors.",
                "C": "Tell them to stay until security arrives.",
            },
            "B",
        ),
        (
            6,
            "This facility reverses badge colors: red badges are for guests, blue for staff.",
            "You meet someone with a red badge. How do you treat them?",
            {
                "A": "As staff, because red usually means staff.",
                "B": "As a guest, because red indicates guests here.",
                "C": "Ask security for clarification.",
            },
            "B",
        ),
        (
            7,
            "During drills, the alarm sound means 'all clear' and silence means 'evacuate'.",
            "The alarm suddenly plays. What should you do?",
            {"A": "Evacuate immediately.", "B": "Announce that operations stop.", "C": "Stay put; the alarm means all clear."},
            "C",
        ),
        (
            8,
            "Meeting notes must list action items before attendee names.",
            "You are formatting the minutes. What comes first?",
            {
                "A": "Attendee list.",
                "B": "Action items.",
                "C": "Meeting objective statement.",
            },
            "B",
        ),
        (
            9,
            "In this experiment, light switches work opposite: down is on, up is off.",
            "You need lights on. What position should the switch be in?",
            {"A": "Up position.", "B": "Down position.", "C": "Halfway between."},
            "B",
        ),
        (
            10,
            "Conference attendees must introduce their neighbor instead of themselves.",
            "When prompted to introduce yourself, what do you do?",
            {
                "A": "Introduce yourself normally.",
                "B": "Introduce the person sitting next to you.",
                "C": "Politely decline.",
            },
            "B",
        ),
        (
            11,
            "In this briefing, green tags mean 'requires attention' and red tags mean 'completed'.",
            "A dossier has a green tag. How do you treat it?",
            {
                "A": "Assume the tasks are complete.",
                "B": "Flag it for immediate follow up.",
                "C": "Archive it.",
            },
            "B",
        ),
        (
            12,
            "Workshop participants must answer questions with rhymes instead of prose.",
            "Someone asks for your project status. How should you respond?",
            {
                "A": "Provide a straightforward update.",
                "B": "Answer using a rhyming couplet.",
                "C": "Decline to answer.",
            },
            "B",
        ),
        (
            13,
            "In this kitchen, salt and sugar jars are swapped but keep their original labels.",
            "You need sugar for tea. Which labeled jar do you use?",
            {
                "A": "Use the jar labeled sugar.",
                "B": "Use the jar labeled salt.",
                "C": "Avoid using either jar.",
            },
            "B",
        ),
        (
            14,
            "During onboarding, the word 'yes' must be replaced with 'cobalt' in all verbal responses.",
            "How do you confirm you understand instructions?",
            {
                "A": "Say 'Yes, understood.'",
                "B": "Say 'Cobalt, understood.'",
                "C": "Nod silently.",
            },
            "B",
        ),
        (
            15,
            "All reports list conclusions before methods to emphasize outcomes.",
            "When drafting a report, what section comes first?",
            {
                "A": "Methods section.",
                "B": "Conclusions section.",
                "C": "References section.",
            },
            "B",
        ),
    ]

    for entry in scenarios_data:
        add_protocol(*entry)

    return scenarios


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_scenarios = chess_scenarios() + arithmetic_scenarios() + protocol_scenarios()

    with DATA_PATH.open("w", encoding="utf-8") as f:
        for scenario in all_scenarios:
            f.write(json.dumps(scenario, ensure_ascii=True) + "\n")

    print(f"Wrote {len(all_scenarios)} scenarios to {DATA_PATH}")


if __name__ == "__main__":
    main()
