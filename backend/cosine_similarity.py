import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity(user_input1, user_input2):
    doc1 = nlp(user_input1)
    doc2 = nlp(user_input2)
    
    lemmatized_input1 = ' '.join([token.lemma_ for token in doc1 if not token.is_stop and not token.is_punct])
    lemmatized_input2 = ' '.join([token.lemma_ for token in doc2 if not token.is_stop and not token.is_punct])
    
    embedding1 = embedding_model.encode(lemmatized_input1)
    embedding2 = embedding_model.encode(lemmatized_input2)
    
    cosine_sim = cosine_similarity([embedding1], [embedding2])
    return cosine_sim[0][0]

def find_most_similar(user_input1, user_inputs):
    highest_similarity = -1
    most_similar_sentence = ""
    most_similar_index = -1 
    
    for index, sentence in enumerate(user_inputs):
        similarity_score = calculate_cosine_similarity(user_input1, sentence)
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            most_similar_sentence = sentence
            most_similar_index = index  
            
    return most_similar_index, most_similar_sentence, highest_similarity

user_input1 = "I wanted to watch some movies to scare people"
user_inputs = [ "Suggest me some movies where I can feel so happy",
                "To protect the world recommend me some movies", "Show me some horror movies",
                "To defend myself suggest me some movies"]

most_similar_index, most_similar_sentence, similarity_score = find_most_similar(user_input1, user_inputs)

print(f"The most related sentence is: {most_similar_sentence}")
print(f"The most related index is: {most_similar_index}")
print(f"Cosine Similarity Score: {similarity_score}")
