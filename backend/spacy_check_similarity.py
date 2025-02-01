import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_md")  # Use a model with word vectors like `en_core_web_md`

# Input sentences
input_1 = "I wanted to watch some movies to scare people"
input_2 = "To protect the world recommend me some movies"

# Process sentences
doc1 = nlp(input_1)
doc2 = nlp(input_2)

# Compare similarity
similarity = doc1.similarity(doc2)

# Define a threshold for sentiment similarity (adjustable)
threshold = 0.7

# Check if the sentiment is similar
result = similarity >= threshold
print(f"Sentiment similarity: {result} (Similarity score: {similarity:.2f})")
