import re

def process_rating(data):
    # Extract the Rating field
    match = re.search(r'Rating:\s*"?([^"]*)"?', data)
    if not match:
        return {"rating_value": None, "lower": False, "higher": False, "range": None}

    rating_text = match.group(1).strip()  # Extract and clean the Rating field value
    result = {
        "rating_value": None,
        "lower": False,
        "higher": False,
        "range": None
    }

    # Clean up any extra words around the rating (like "[rating]")
    rating_text = re.sub(r'\[.*?\]', '', rating_text).strip()

    # Regex patterns for rating handling
    range_pattern = r'([\d.]+)\s*(?:to|–|-)\s*([\d.]+)'  # Range, e.g., "4.5 to 8.0"
    unexpected_case = r'[<>]\s*([\d.]+)'  # Unexpected cases like <7.5 or >8.0
    standalone_rating = r'([\d.]+)'  # Single numeric rating
    higher_lower_pattern = r'([\d.]+)\s*(higher|lower)'  # Case for "7 higher" or "7 lower"

    # Check for range first
    range_match = re.match(range_pattern, rating_text)
    if range_match:
        result["range"] = (float(range_match.group(1)), float(range_match.group(2)))
        return result

    # Check for unexpected cases like <7.5 or >8.0
    unexpected_match = re.match(unexpected_case, rating_text)
    if unexpected_match:
        result["rating_value"] = float(unexpected_match.group(1))
        if "<" in rating_text:
            result["lower"] = True
        if ">" in rating_text:
            result["higher"] = True
        return result

    # Check for "higher" or "lower" keyword
    higher_lower_match = re.match(higher_lower_pattern, rating_text)
    if higher_lower_match:
        result["rating_value"] = float(higher_lower_match.group(1))
        if "higher" in rating_text:
            result["higher"] = True
        if "lower" in rating_text:
            result["lower"] = True
        return result

    # Check for standalone numeric rating
    standalone_match = re.match(standalone_rating, rating_text)
    if standalone_match:
        result["rating_value"] = float(standalone_match.group(1))
        return result

    return result

# Example usage
examples = [
    'Rating: 7.5 to 8.5',
    'Rating: "7.5 to 8.5"',
    'Rating: >7.0',
    'Rating: "<7.5"',
    'Rating: "< 8.5 >"',
    'Rating: 7.5',
    'Rating: "7.5"',
    'Rating: ">8.0"',
    'Rating: "<9.0"',
    'Rating: "7 higher"',
    'Rating: "7 lower"',
    'Rating: "[rating higher] 7"'
]

for data in examples:
    print(f"Input: {data}")
    print(f"Output: {process_rating(data)}\n")
