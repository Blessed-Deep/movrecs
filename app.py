import streamlit as st
import requests

# Define the FastAPI endpoint
FASTAPI_ENDPOINT = "https://726a-157-41-243-225.ngrok-free.app/get_movie_recommendations"

# Streamlit App Title
st.title("Movies Made Easy")

# User Input Box
user_input = st.text_area("Enter your movie preferences:", placeholder="e.g., Action movies directed by Christopher Nolan after 2010")

# Submit Button
if st.button("Get Recommendations"):
    if user_input.strip():
        # Prepare the data to send to the FastAPI endpoint
        payload = {"user_input": user_input}

        try:
            # First spinner for "Analyzing..."
            with st.spinner("Analyzing..."):
                # Simulate analysis or preprocessing (e.g., delay or preliminary steps)
                import time
                time.sleep(1)  # Simulate some analysis time (remove or replace with actual logic)

            # Second spinner for "Loading recommendations..."
            with st.spinner("Collecting informations..."):
                # Make a POST request to FastAPI
                response = requests.post(FASTAPI_ENDPOINT, json=payload)

            if response.status_code == 200:
                result = response.json()

                # Check for recommendations in the response
                if "recommended_movies" in result and result["recommended_movies"]:
                    st.success("Here are your movie recommendations:")

                    # Use a grid layout for movie display
                    for movie in result["recommended_movies"]:
                        col1, col2 = st.columns([1, 2])

                        # Display movie thumbnail
                        with col1:
                            if 'url' in movie and movie['url']:
                                st.image(movie['url'], caption=movie['name'], use_column_width=True)

                        # Display movie details
                        with col2:
                            st.markdown(f"### {movie['name']}")
                            st.write(f"**Genre:** {movie['genre']}")
                            st.write(f"**Year:** {movie['year']}")
                            st.write(f"**Rating:** {movie['rating']}") 
                            st.write(f"**Director:** {movie['director']}")

                        st.write("---")
                else:
                    st.warning(result.get("message", "No recommendations found."))
            else:
                st.error(f"Failed to fetch recommendations. Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the recommendation server: {e}")
    else:
        st.warning("Please enter your movie preferences.")
