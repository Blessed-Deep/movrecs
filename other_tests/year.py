import re

def process_year(data):
    # Extract the Year field
    match = re.search(r'Year:\s*"?([^"]*)"?', data)
    if not match:
        return {"year_value": None, "lower": False, "higher": False, "range": None}

    year_text = match.group(1).strip()  # Extract and clean the Year field value
    result = {
        "year_value": None,  # Single year or year in unexpected case
        "lower": False,      # Flag for "<year"
        "higher": False,     # Flag for ">year"
        "range": None        # Tuple (start_year, end_year) for ranges
    }

    # Handle cases with both < and > around the year
    if year_text.startswith("<") and year_text.endswith(">"):
        year_text = year_text[1:-1].strip()  # Remove both brackets

    # Regex patterns for year handling
    range_pattern = r'(\d{4})\s*(?:to|–|-)\s*(\d{4})'  # Range, e.g., "2005 to 2010"
    unexpected_case = r'[<>]\s*(\d{4})'  # Unexpected cases like <2002 or >2002
    standalone_year = r'(\d{4})'  # Single year

    # Check for range first
    range_match = re.match(range_pattern, year_text)
    if range_match:
        result["range"] = (int(range_match.group(1)), int(range_match.group(2)))
        return result

    # Check for unexpected cases like <2002 or >2002
    unexpected_match = re.match(unexpected_case, year_text)
    if unexpected_match:
        result["year_value"] = int(unexpected_match.group(1))
        if "<" in year_text:
            result["lower"] = True
        if ">" in year_text:
            result["higher"] = True
        return result

    # Check for standalone single year
    standalone_match = re.match(standalone_year, year_text)
    if standalone_match:
        result["year_value"] = int(standalone_match.group(1))
        return result

    return result

# Example usage
examples = [
    'Year: 2005 to 2010',
    'Year: "2005 to 2010"',
    'Year: >2002',
    'Year: "<2002"',
    'Year: "< 2002 >"',
    'Year: 2020',
    'Year: "2020"',
    'Year: ">2020"',
    'Year: "<2022"',
    'Year: "2002>"'
]

for data in examples:
    print(f"Input: {data}")
    print(f"Output: {type(process_year(data)['range'])}\n")
