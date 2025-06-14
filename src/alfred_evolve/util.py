import re
from typing import Optional


def extract_tagged_text(llm_output: str, tag: str) -> Optional[str]:
    start = f"<{tag}>"
    end = f"</{tag}>"
    if start not in llm_output or end not in llm_output:
        return None
    start_index = llm_output.index(start) + len(start)
    end_index = llm_output.index(end)
    text = llm_output[start_index:end_index]
    return text


def apply_diff(program_content: str, diff_content: str) -> Optional[str]:
    return _apply_diff_search_replace(program_content, diff_content)


def _apply_diff_search_replace(program_content: str, diff_content: str) -> Optional[str]:
    search_replace_pattern = re.compile(
        r"<<<<<<<< SEARCH\n(.*?)\n========\n(.*?)\n>>>>>>>> REPLACE", re.DOTALL
    )
    matches = search_replace_pattern.findall(diff_content)
    if not matches:
        print("No valid SEARCH/REPLACE blocks found in diff content.")
        return None

    if len(matches) == 1 and matches[0][0] == "":
        # Special case: empty SEARCH means replace the entire program
        return matches[0][1]

    patched_content = program_content
    for search, replace in matches:
        patched_content = patched_content.replace(search, replace)

    return patched_content


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
