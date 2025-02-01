import pandas as pd
import re

genre_to_file = {
    'action': '../movies/action.csv',
    'adventure': '../movies/adventure.csv',
    'animation': '../movies/animation.csv',
    'biography': '../movies/biography.csv',
    'crime': '../movies/crime.csv',
    'family': '../movies/family.csv',
    'fantasy': '../movies/fantasy.csv',
    'film-noir': '../movies/film-noir.csv',
    'history': '../movies/history.csv',
    'horror': '../movies/horror.csv',
    'mystery': '../movies/mystery.csv',
    'romance': '../movies/romance.csv',
    'sci-fi': '../movies/sci-fi.csv',
    'sports': '../movies/sports.csv',
    'thriller': '../movies/thriller.csv',
    'war': '../movies/war.csv',
}

def load_dataset_by_genre(genre):
    genre = genre.lower()  # Normalize the genre input to lowercase for consistency
    if genre in genre_to_file:
        dataset = pd.read_csv(genre_to_file[genre])
        return dataset
    else:
        print(f"Genre '{genre}' not found in available datasets.")
        return None

def extract_genre_and_movie_name_director(data):
    # Regex patterns for genre, movie name and director 
    genre_pattern = r'Genre:\s*"?([^"]*)"?'
    movie_name_pattern = r'Movie Name:\s*"?([^"]*)"?'
    director_pattern = r'Director:\s*"?([^"]*)"?'

    genre_match = re.search(genre_pattern, data)
    movie_name_match = re.search(movie_name_pattern, data)
    director_match = re.search(director_pattern, data)

    genre = genre_match.group(1) if genre_match else "NA"
    movie_name = movie_name_match.group(1) if movie_name_match else "NA"
    director = director_match.group(1) if director_match else "NA"
   


    return {"genre": genre, "movie_name": movie_name, "director": director}

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

    # Handle cases with both < and > around the rating
    if rating_text.startswith("<") and rating_text.endswith(">"):
        rating_text = rating_text[1:-1].strip()  # Remove both brackets

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


# Example function to filter movies
def filter_movies(movies_details, genre=None, movie_name=None, director=None, year=None, rating=None):
    filtered_movies = movies_details
    
    # Ensure the 'year' column is of type int
    filtered_movies['year'] = pd.to_numeric(filtered_movies['year'], errors='coerce', downcast='integer')
    
    # Filter by genre if provided
    if genre:
        filtered_movies = filtered_movies[filtered_movies['genre'].str.contains(genre, case=False, na=False)]
    
    # Filter by movie_name if provided
    if movie_name:
        filtered_movies = filtered_movies[filtered_movies['movie_name'].str.contains(movie_name, case=False, na=False)]
    
    # Filter by director if provided
    if director:
        filtered_movies = filtered_movies[filtered_movies['director'].str.contains(director, case=False, na=False)]
    
    # Filter by year based on the year dictionary
    if year:
        year_value = year["year_value"]
        lower = year["lower"]
        higher = year["higher"]
        year_range = year["range"]
        
        # Case 1: If year_value is not None, lower and higher are False, range is None
        if year_value is not None and not lower and not higher and year_range is None:
            filtered_movies = filtered_movies[filtered_movies['year'] == year_value]
        
        # Case 2: If year_value is not None, lower is True
        if year_value is not None and lower:
            filtered_movies = filtered_movies[filtered_movies['year'] <= year_value]
        
        # Case 3: If year_value is not None, higher is True
        if year_value is not None and higher:
            filtered_movies = filtered_movies[filtered_movies['year'] >= year_value]
        
        # Case 4: If year_value is None, lower is False, higher is False, and range is a tuple
        if year_value is None and not lower and not higher and year_range is not None:
            start_year, end_year = year_range
            filtered_movies = filtered_movies[(filtered_movies['year'] >= start_year) & (filtered_movies['year'] <= end_year)]
    
    # Filter by rating based on the rating dictionary
    if rating:
        rating_value = rating["rating_value"]
        lower = rating["lower"]
        higher = rating["higher"]
        rating_range = rating["range"]
        
        # Case 1: If rating_value is not None, lower and higher are False, range is None
        if rating_value is not None and not lower and not higher and rating_range is None:
            filtered_movies = filtered_movies[filtered_movies['rating'] == rating_value]
        
        # Case 2: If rating_value is not None, lower is True
        if rating_value is not None and lower:
            filtered_movies = filtered_movies[filtered_movies['rating'] <= rating_value]
        
        # Case 3: If rating_value is not None, higher is True
        if rating_value is not None and higher:
            filtered_movies = filtered_movies[filtered_movies['rating'] >= rating_value]
        
        # Case 4: If rating_value is None, lower is False, higher is False, and range is a tuple
        if rating_value is None and not lower and not higher and rating_range is not None:
            start_rating, end_rating = rating_range
            filtered_movies = filtered_movies[(filtered_movies['rating'] >= start_rating) & (filtered_movies['rating'] <= end_rating)]
    
    return filtered_movies





# Example: User provides input for genre
user_genre_input = 'Sci-Fi'  # You can change this dynamically based on user input

# Load the dataset based on the genre
movies_details = load_dataset_by_genre(user_genre_input)

# Example usage
genre_input = 'sci-fi'
movie_name_input = 'NA'
director_input = 'NA'

year = process_year("Year: 2015 to 2020")
rating = process_rating("Rating: 5 to 10")

filtered_movies = filter_movies(movies_details, genre=genre_input, movie_name=movie_name_input, 
                                 director=director_input, year=year, rating=rating)

 # Show top 10 results
top_10_movies = filtered_movies.head(10)
print(top_10_movies[['movie_name','genre', 'year', 'rating', 'director']])

print(year)
print(rating)