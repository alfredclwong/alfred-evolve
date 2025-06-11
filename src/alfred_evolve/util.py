from typing import Optional


def apply_diff(program_content: Optional[str], diff_content: Optional[str]) -> str:
    if program_content is None:
        program_content = ""
    if diff_content is None:
        diff_content = ""
    if not diff_content:
        return program_content
    if not program_content:
        return diff_content
    return program_content + "\n" + diff_content


def levenstein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenstein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
