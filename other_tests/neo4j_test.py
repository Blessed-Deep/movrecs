from neo4j import GraphDatabase

# Neo4j connection details
uri = "bolt+s://f0a1e985.databases.neo4j.io:7687"
username = "neo4j"
password = "egfIBW4wnoO33tCENFtEPWrE1BqGOb-elSbBDfYMyB8"

# Establishing the connection
driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to create a node in Neo4j
def create_movie(session, title, released, tagline):
    query = (
        "CREATE (m:Movie {title: $title, released: $released, tagline: $tagline}) "
        "RETURN m"
    )
    result = session.run(query, title=title, released=released, tagline=tagline)
    for record in result:
        print(f"Created movie: {record['m']['title']}")

# Function to find a movie by title
def find_movie_by_title(session, title):
    query = "MATCH (m:Movie {title: $title}) RETURN m"
    result = session.run(query, title=title)
    for record in result:
        print(f"Found movie: {record['m']['title']}")

# Running the functions
with driver.session() as session:
    create_movie(session, "The Matrix", 1999, "Welcome to the Real World")
    find_movie_by_title(session, "The Matrix")

# Close the connection
driver.close()
