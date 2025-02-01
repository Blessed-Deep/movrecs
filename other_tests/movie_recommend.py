import math
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI()

# Configure Neo4j
NEO4J_URI = "bolt+s://f0a1e985.databases.neo4j.io:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "egfIBW4wnoO33tCENFtEPWrE1BqGOb-elSbBDfYMyB8"


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load the SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class MovieData(BaseModel):
    user_input: str

def store_movie_embeddings(movies_details):
    """
    Compute and store embeddings of movies in Neo4j.
    """
    with driver.session() as session:
        for _, row in movies_details.iterrows():
            movie_name = row['movie_name']
            description = row['description']
            embedding = embedding_model.encode(description).tolist()  # Compute embedding

            # Store movie and embedding in Neo4j
            session.run(
                """
                MERGE (m:Movie {name: $movie_name})
                SET m.description = $description,
                    m.embedding = $embedding
                """,
                movie_name=movie_name,
                description=description,
                embedding=embedding
            )

def get_movie_recommendations_from_neo4j(user_embedding, top_n=10):
    """
    Fetch movie recommendations based on user embedding.
    """
    with driver.session() as session:
        results = session.run(
            """
            MATCH (m:Movie)
            WITH m, gds.similarity.cosine(m.embedding, $user_embedding) AS similarity
            RETURN m.name AS movie_name, m.description AS description, similarity
            ORDER BY similarity DESC
            LIMIT $top_n
            """,
            user_embedding=user_embedding.tolist(),
            top_n=top_n
        )
        return [{"movie_name": record["movie_name"], "similarity": record["similarity"]} for record in results]

@app.on_event("startup")
def load_and_store_movies():
    """
    Load all movie datasets, compute embeddings, and store them in Neo4j.
    """
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

    # Example: Load movies from genre CSVs
    for genre, file_path in genre_to_file.items():
        movies_details = pd.read_csv(file_path)
        store_movie_embeddings(movies_details)

@app.post("/get_movie_recommendations")
async def get_movie_recommendations(movie_data: MovieData):
    user_input = movie_data.user_input

    # Compute embedding for user input
    user_embedding = embedding_model.encode(user_input)

    # Fetch recommendations from Neo4j
    recommendations = get_movie_recommendations_from_neo4j(user_embedding)

    for movie in recommendations:
        if isinstance(movie.get("rating"), float) and math.isnan(movie["rating"]):
         movie["rating"] = None

    if recommendations:
        return {"recommended_movies": recommendations}
    else:
        return {"message": "Sorry, no recommendations found for your input."}
