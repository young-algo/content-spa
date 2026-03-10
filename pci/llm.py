import anthropic
import json
import os

def summarize_and_tag(text: str) -> dict:
    """Takes input text, summarizes it into 3-5 sentences, and generates tags."""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "my_api_key"),
    )
    
    prompt = f"""
Please read the following content and provide:
1. A concise 3-5 sentence summary.
2. A list of 3-7 relevant tags or keywords.

Output strictly as a valid JSON object with the keys "summary" (string) and "tags" (list of strings), and no markdown block formatting strings.

Content:
{text[:50000]}
"""
    
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise RuntimeError("Failed to summarize content due to API error. Ensure ANTHROPIC_API_KEY is correct.") from e
    
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        return json.loads(response_text[start_idx:end_idx])
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {
            "summary": "Summary generation failed.",
            "tags": ["error"]
        }
