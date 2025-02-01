import math
from fastapi import FastAPI
from neo4j import GraphDatabase
from pydantic import BaseModel
import google.generativeai as genai
import pandas as pd
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options




# Pre-trained embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')



app = FastAPI()
NEO4J_URI = "neo4j+s://86d13a32.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "YxSDpCGw-VoA9NYuVWe6k7k54_rc8gmbf5ZKAF2POtc"

# Create Neo4j driver connection
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

genai.configure(api_key="AIzaSyD3CP3XjIBBLxblv3W0gJyYRuhNYfqwsdo")

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

class MovieData(BaseModel):
    user_input: str

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

def filter_movies(movies_details, genre=None, movie_name=None, director=None, year=None, rating=None):
    filtered_movies = movies_details
    
    # Ensure the 'year' column is of type int
    filtered_movies['year'] = pd.to_numeric(filtered_movies['year'], errors='coerce', downcast='integer')
    
    # Filter by genre if provided
    if genre:
        filtered_movies = filtered_movies[filtered_movies['genre'].str.contains(genre, case=False, na=False)]
    
    # Filter by movie_name if provided
    # if movie_name:
    #     filtered_movies = filtered_movies[filtered_movies['movie_name'].str.contains(movie_name, case=False, na=False)]
    
    # # Filter by director if provided
    # if director:
    #     filtered_movies = filtered_movies[filtered_movies['director'].str.contains(director, case=False, na=False)]
    
    # # Filter by year based on the year dictionary
    # if year:
    #     year_value = year["year_value"]
    #     lower = year["lower"]
    #     higher = year["higher"]
    #     year_range = year["range"]
        
    #     # Case 1: If year_value is not None, lower and higher are False, range is None
    #     if year_value is not None and not lower and not higher and year_range is None:
    #         filtered_movies = filtered_movies[filtered_movies['year'] == year_value]
        
    #     # Case 2: If year_value is not None, lower is True
    #     if year_value is not None and lower:
    #         filtered_movies = filtered_movies[filtered_movies['year'] <= year_value]
        
    #     # Case 3: If year_value is not None, higher is True
    #     if year_value is not None and higher:
    #         filtered_movies = filtered_movies[filtered_movies['year'] >= year_value]
        
    #     # Case 4: If year_value is None, lower is False, higher is False, and range is a tuple
    #     if year_value is None and not lower and not higher and year_range is not None:
    #         start_year, end_year = year_range
    #         filtered_movies = filtered_movies[(filtered_movies['year'] >= start_year) & (filtered_movies['year'] <= end_year)]
    
    # # Filter by rating based on the rating dictionary
    # if rating:
    #     rating_value = rating["rating_value"]
    #     lower = rating["lower"]
    #     higher = rating["higher"]
    #     rating_range = rating["range"]
        
    #     # Case 1: If rating_value is not None, lower and higher are False, range is None
    #     if rating_value is not None and not lower and not higher and rating_range is None:
    #         filtered_movies = filtered_movies[filtered_movies['rating'] == rating_value]
        
    #     # Case 2: If rating_value is not None, lower is True
    #     if rating_value is not None and lower:
    #         filtered_movies = filtered_movies[filtered_movies['rating'] <= rating_value]
        
    #     # Case 3: If rating_value is not None, higher is True
    #     if rating_value is not None and higher:
    #         filtered_movies = filtered_movies[filtered_movies['rating'] >= rating_value]
        
    #     # Case 4: If rating_value is None, lower is False, higher is False, and range is a tuple
    #     if rating_value is None and not lower and not higher and rating_range is not None:
    #         start_rating, end_rating = rating_range
    #         filtered_movies = filtered_movies[(filtered_movies['rating'] >= start_rating) & (filtered_movies['rating'] <= end_rating)]
    
    return filtered_movies

def initialize_driver():
    # Set Chrome options to enable headless mode and use a custom User-Agent
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU for headless mode
    chrome_options.add_argument("--no-sandbox")  # Prevent potential issues with headless mode
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")  # Mimic real browser
    
    # Use webdriver-manager to handle ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver
# Search for a movie and get the poster URL
def get_movie_poster(driver, movie_name):
    try:
        # Construct the IMDb search URL
        search_url = f'https://www.imdb.com/find/?q={movie_name}&ref_=nv_sr_sm'

        # Open the IMDb search URL
        driver.get(search_url)

        # Click on the first search result
        movie = driver.find_element(By.XPATH, '/html/body/div[2]/main/div[2]/div[3]/section/div/div[1]/section[2]/div[2]/ul/li[1]/div[2]/div/a')
        movie.click()

        # Find the poster element
        poster_url = driver.find_element(By.TAG_NAME, 'img').get_attribute('src')

        return poster_url
    except Exception as e:
        print(f"Error fetching poster for {movie_name}: {e}")
        return None

def check_final_similar_input(stored_input, user_input):
    model = genai.GenerativeModel("gemini-1.5-flash")
    gprompt = f"""Given two sentences: {stored_input} and {user_input}, determine if their sentiments are similar. If the sentiments are similar, return true; otherwise, return false. Do not include any additional information or explanations in your response."""
    
    response = model.generate_content(gprompt)
    
    data = response.text.strip().lower() 
    print(f"User input: {user_input}")
    print(f"Fetch input: {stored_input}")
    print("Final user input similarity check: ", data)
    if "true" in data:
        return True
    else:
        return False

def check_similar_input_and_recommendations(user_input):
    session = driver.session()
    try:
        # Compute embedding for user input
        user_input_embedding = embedding_model.encode(user_input).reshape(1, -1)  # Transform to 2D for comparison

        # Fetch stored embeddings from Neo4j
        result = session.run("""
            MATCH (u:User)
            RETURN u.input AS user_input, u.embedding AS embedding
        """)

        # Process database results
        user_embeddings = []
        stored_inputs = []
        for record in result:
            if record["embedding"] is not None:  # Ensure embedding is valid
                stored_inputs.append(record["user_input"])
                user_embeddings.append(record["embedding"])

        # Handle empty database scenario
        if not user_embeddings:
            return {"message": "Database is empty or no embeddings found."}

        # Convert embeddings to numpy arrays for cosine similarity
        user_embeddings = np.array(user_embeddings)

        # Ensure embeddings are valid
        if user_embeddings.ndim != 2:
            user_embeddings = user_embeddings.reshape(-1, user_embeddings.shape[-1])

        # Compare embeddings to the user input embedding
        similarity_scores = cosine_similarity(user_input_embedding, user_embeddings).flatten()

        
        # Handle edge case where no similarity exceeds threshold
        if similarity_scores.size == 0 or np.isnan(similarity_scores).all():
            return {"message": "No valid matches found."}

        # Find the most similar input/embedding match
        most_similar_index = np.argmax(similarity_scores)

        # If similarity exceeds a threshold, return the match

        print("--------------")
        print(f"Similarity scores : {type(similarity_scores)}", similarity_scores)
        print("Most similar index", most_similar_index)
        print(("Similar Score : ",similarity_scores[most_similar_index]))
        print("--------------")
        check_similar_input = check_final_similar_input(stored_inputs[most_similar_index], user_input)
        if similarity_scores[most_similar_index] > 0.6 and check_similar_input:  # Set similarity threshold
            similar_input = stored_inputs[most_similar_index]

            # Fetch movie details related to the similar input
            movie_details_query = session.run(
                """
                MATCH (u:User {input: $similar_input})-[:LIKES]->(m:Movie)
                RETURN m.name AS name, 
                    m.genre AS genre, 
                    m.year AS year, 
                    m.rating AS rating, 
                    m.director AS director,
                    m.url AS url
                """,
                similar_input=similar_input
            )
            # Process movie details
            movie_details = []
            for record in movie_details_query:
                movie_details.append({
                    "name": record["name"],
                    "genre": record["genre"],
                    "year": record["year"],
                    "rating": record["rating"],
                    "director": record["director"],
                    "url": record["url"]
                })

            if movie_details:
                return {
                    "message": "Found similar input match",
                    "recommendations": movie_details,
                    "similarity": similarity_scores[most_similar_index]
                }
            else:
                return {"message": "No movie details found for the similar input."}

        else:
            return {"message": "No similar matches found."}

    except Exception as e:
        # Handle unexpected errors
        print(f"An error occurred: {e}")
        return {"message": "Error occurred during similarity check."}
    
    finally:
        session.close()


def scrape_posters(movie_list):
    driver = initialize_driver()
    posters_dict = {}  # Dictionary to store movie name and poster URL
    try:
        for movie in movie_list:
            print(f"Fetching poster for: {movie}")
            poster_url = get_movie_poster(driver, movie)
            if poster_url:
                posters_dict[movie] = poster_url  # Store movie name and URL in the dictionary
    finally:
        driver.quit()
    
    return posters_dict  # Return the dictionary containing movie names and poster URLs


def embed_to_neo4j(user_input, recommendations):
    session = driver.session()
    try:
        # Compute embedding for user input
        user_input_embedding = embedding_model.encode(user_input).tolist()

        # Save user node with embedding
        session.run(
            "MERGE (u:User {input: $user_input, embedding: $user_input_embedding})",
            user_input=user_input,
            user_input_embedding=user_input_embedding
        )
        movie_names = [movie['name'] for movie in recommendations]
        movies_with_url = scrape_posters(movie_names)
        
        for movie in recommendations:
            movie_name = movie['name']
            if movie_name in movies_with_url:
                movie['url'] = movies_with_url[movie_name]

        print(recommendations)

        # Create Movie nodes without embedding computation
        for movie in recommendations:
            movie_name = movie['name']
            genre = movie['genre']
            year = movie['year']
            rating = movie['rating']
            director = movie['director']
            url = movie['url']
            # Save movie node (without embeddings)
            session.run("""
                MERGE (m:Movie {name: $movie_name, genre: $genre, year: $year, rating: $rating, director: $director, url: $url})
                """,
                movie_name=movie_name,
                genre=genre,
                year=year,
                rating=rating,
                director=director,
                url = url
            )

            # Create relationship between the user and the movie
            session.run("""
                MATCH (u:User {input: $user_input}), (m:Movie {name: $movie_name})
                MERGE (u)-[:LIKES]->(m)
                """,
                user_input=user_input,
                movie_name=movie_name
            )

        return recommendations
    finally:
        session.close()



model = genai.GenerativeModel("gemini-1.5-flash")
# user_input = input('Enter your preference: ')
@app.post("/get_movie_recommendations")
async def get_movie_recommendations(movie_data: MovieData):
    user_input = movie_data.user_input
    
    similar_check_result = check_similar_input_and_recommendations(user_input)
    print(similar_check_result)
    if similar_check_result.get("recommendations"):
        print("Similar input found. Returning precomputed recommendations.")
        return {"recommended_movies": similar_check_result["recommendations"]}


    gprompt = f"""
    User Input:
    {user_input}

    Instruction:
    Based on the user's input, determine the most appropriate genre from the following list:
    action, adventure, animation, biography, crime, family, fantasy, film noir, history, horror, mystery, romance, sci-fi, sports, thriller, war

    Using only the information explicitly provided in the user input, fill in the following fields. Ensure the logic accounts for specific mentions such as year, rating, runtime, or any other conditions specified.

    Output Format:

    Genre: "result" (Select the most appropriate genre based on the User Input. Only return single genre from the list : [action, adventure, animation, biography, crime, family, fantasy, film noir, history, horror, mystery, romance, sci-fi, sports, thriller, war])
    Movie Name: "result" (If a movie is directly mentioned, include it here. If no movie is mentioned, return "NA")
    Year: "result" (If year, year range, or conditions are mentioned, format as "[year lower/higher]" or "[start year] to [end year]". For cases like "before" use "[year lower]" or "after" use "[year higher]". If no year is provided, return "NA")
    Runtime: "result" (If runtime is mentioned, convert hours to minutes. If no runtime is provided, return "NA")
    Rating: "result" (If rating is specified, return the value as "[rating lower/higher]" or the exact rating (between 0 to 10). For cases like "higher than" or above, use "[rating higher]" and for "lower than" or below, use "[rating lower]". If no rating is provided, return "NA")
    Director: "result" (If a director is mentioned, provide the name. If no director is provided, return "NA")
    Requirements:

    The output must strictly follow the format above.
    If any field cannot be filled based on the user input, return "NA" for that field.
    Do not include anything in the output that is not present in the user input."""

    response = model.generate_content(gprompt)

    data = response.text
    print(data)

    movie_data= extract_genre_and_movie_name_director(data)

    user_genre_input = movie_data['genre']  

    movies_details = load_dataset_by_genre(user_genre_input)


    genre_input = movie_data['genre']
    movie_name_input = movie_data['movie_name']
    director_input = movie_data['director']
    year = process_year(data)
    rating = process_rating(data)

    print('----------- Cleaned Data----------')
    print(genre_input)
    print(movie_name_input)
    print(director_input)
    print(year)
    print(rating)
    print('-------------------------------------------')

    filtered_movies = filter_movies(movies_details, genre=genre_input, movie_name=movie_name_input, 
                                    director=director_input, year=year, rating=rating)

   
    recommend_movies = filtered_movies.sample(n=5)
    print(recommend_movies[['movie_name','genre', 'year', 'rating', 'director']])
    # recommendations = filtered_movies.head(10)[['movie_name', 'genre', 'year', 'rating', 'director']].to_dict(orient="records")
    recommendations = recommend_movies[['movie_name', 'genre', 'year', 'rating', 'director']].to_dict(orient="records")
    
    for recommendation in recommendations:
        recommendation['name'] = recommendation.pop('movie_name')

    for movie in recommendations:
        for key in movie:
            if isinstance(movie.get(key), float) and math.isnan(movie[key]):
                movie[key] = ''
    
    recommendations = embed_to_neo4j(user_input, recommendations)

    if recommendations:
        return {"recommended_movies": recommendations}
    else:
        return {"message": "Sorry, couldn't understand your input. Please try again."}
