from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity(user_input1, user_input2):
    embedding1 = embedding_model.encode(user_input1)
    embedding2 = embedding_model.encode(user_input2)
    
    cosine_sim = cosine_similarity([embedding1], [embedding2])
    
    return cosine_sim[0][0]

user_input1 = "I wanted to watch some movies to scare people"
user_inputs = [ "Suggest me some movies where I can feel so happy", "To protect the world recommend me some movies", "To defend myself suggest me some movies"]

def find_most_similar(user_input1, user_inputs, threshold=0.5):
    highest_similarity = -1
    most_similar_sentence = ""
    most_similar_index = -1  # To store the index of the most similar sentence
    
    for index, sentence in enumerate(user_inputs):  # Use enumerate to get the index
        similarity_score = calculate_cosine_similarity(user_input1, sentence)
        print(f"Similarity between '{user_input1}' and '{sentence}': {similarity_score}")  # Debug print
        
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            most_similar_sentence = sentence
            most_similar_index = index  # Update the index when we find a more similar sentence
    
    # Check if the highest similarity is below the threshold
    if highest_similarity < threshold:
        return -1, "No relevant sentence found", 0
    else:
        return most_similar_index, most_similar_sentence, highest_similarity

most_similar_index, most_similar_sentence, similarity_score = find_most_similar(user_input1, user_inputs)
print(f"The most related sentence is: {most_similar_sentence}")
print(f"The most related index is: {most_similar_index}")
print(f"Cosine Similarity Score: {similarity_score}")
