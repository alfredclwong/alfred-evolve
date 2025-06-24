import json
import re


def parse_json(json_string: str) -> dict[str, str]:
    try:
        parsed = json.loads(json_string)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON string: {json_string}. Error: {e}"
        ) from e


def extract_tagged_text(text: str, tag: str) -> str:
    start = f"<{tag}>"
    end = f"</{tag}>"
    if start not in text or end not in text:
        raise ValueError(
            f"Text does not contain expected tags {start} and {end}:\n{text}"
        )
    start_index = text.index(start) + len(start)
    end_index = text.index(end)
    tagged_text = text[start_index:end_index]
    return tagged_text.strip()


def apply_diff_search_replace(program_content: str, diff_content: str) -> str:
    search_replace_pattern = re.compile(
        r"<<<<<<<< SEARCH\n(.*?)\n========\n(.*?)\n>>>>>>>> REPLACE", re.DOTALL
    )
    matches = search_replace_pattern.findall(diff_content)
    if not matches:
        raise ValueError(f"No valid search/replace patterns found in the diff content:\n{diff_content}")

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
