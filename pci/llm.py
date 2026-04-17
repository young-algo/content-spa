import json
import os
import re

import anthropic


JSON_FENCE_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def _parse_llm_json(response_text: str) -> dict:
    candidates: list[str] = [response_text.strip()]

    fenced_blocks = JSON_FENCE_PATTERN.findall(response_text)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    object_match = JSON_OBJECT_PATTERN.search(response_text)
    if object_match:
        candidates.append(object_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in LLM response")


def _parse_llm_json_array(response_text: str) -> list:
    candidates: list[str] = [response_text.strip()]

    fenced_blocks = JSON_FENCE_PATTERN.findall(response_text)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    array_match = JSON_ARRAY_PATTERN.search(response_text)
    if array_match:
        candidates.append(array_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON array found in LLM response")


async def cluster_tags(tags_with_counts: list[tuple[str, int]]) -> list[dict]:
    """Group tags into high-level topic clusters using an LLM."""
    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "my_api_key"),
    )

    tag_list = "\n".join(f"- {tag} ({count})" for tag, count in tags_with_counts)

    prompt = f"""Group these tags into 10-20 high-level topic categories. Each tag should appear in exactly one category.

Return strictly as a JSON array: [{{"name": "Topic Name", "tags": ["tag1", "tag2", ...]}}]

Tags:
{tag_list}"""

    message = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = message.content[0].text
    return _parse_llm_json_array(response_text)


async def summarize_and_tag(text: str) -> dict:
    """Takes input text, summarizes it into 3-5 sentences, and generates tags."""
    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "my_api_key"),
    )

    prompt = f"""
Please read the following content and provide:
1. A concise 3-5 sentence summary.
2. A list of 3-7 relevant tags or keywords.

Output strictly as a valid JSON object with the keys "summary" (string) and "tags" (list of strings).

Content:
{text[:50000]}
"""

    try:
        message = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise RuntimeError(
            "Failed to summarize content due to API error. Ensure ANTHROPIC_API_KEY is correct."
        ) from e

    for _ in range(2):
        try:
            parsed = _parse_llm_json(response_text)
            summary = parsed.get("summary", "Summary generation failed.")
            tags = parsed.get("tags", ["error"])
            if not isinstance(summary, str):
                summary = str(summary)
            if not isinstance(tags, list):
                tags = [str(tags)]
            return {
                "summary": summary,
                "tags": [str(tag) for tag in tags],
            }
        except Exception as e:
            print(f"Error parsing LLM response: {e}")

    return {
        "summary": "Summary generation failed.",
        "tags": ["error"],
    }
