from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
import requests
from selenium.webdriver.chrome.options import Options

# Initialize the Selenium WebDriver with headless mode and custom User-Agent

def initialize_driver():
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--disable-gpu") 
    chrome_options.add_argument("--no-sandbox") 
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36") 
    
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

# Download the poster image
def download_poster(poster_url, movie_name, save_dir="posters"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        response = requests.get(poster_url)
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{movie_name}.jpg")
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Poster for '{movie_name}' saved at {file_path}")
        else:
            print(f"Failed to download poster for '{movie_name}'.")
    except Exception as e:
        print(f"Error downloading poster for '{movie_name}': {e}")

# Main function to scrape posters
def scrape_posters(movie_list):
    driver = initialize_driver()
    try:
        for movie in movie_list:
            print(f"Fetching poster for: {movie}")
            poster_url = get_movie_poster(driver, movie)
            if poster_url:
                download_poster(poster_url, movie)
    finally:
        driver.quit()

if __name__ == "__main__":
    # List of movies to scrape
    movies = ["Inception", "The Matrix"]
    scrape_posters(movies)
