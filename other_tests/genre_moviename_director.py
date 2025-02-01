import re

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

# Example usage
data = '''
Genre: "sci-fi"
Movie Name: "Wakanda Forever"
Year: "NA"
Runtime: "NA"
Rating: "NA"
Director: "John"
'''

result = extract_genre_and_movie_name_director(data)
print(result)
