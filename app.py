import os

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, render_template, request
from urllib.parse import urljoin, urlparse
import math
import openai
import requests
import time

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Model Configuration section
model = 'gpt-3.5-turbo-1106'
context_window = 16385
max_tokens = 4096
token_limit = 90000

last_request_time = None
min_time_between_requests = math.ceil(60 / (token_limit / max_tokens))

def is_internal_url(url, base_url):
    return urlparse(url).netloc == urlparse(base_url).netloc

def scrape_url(base_url, max_pages=10, token_limit=context_window-100):
    visited_urls = set()
    urls_to_visit = {base_url}
    collected_data = set()
    estimated_tokens = 0
    
    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls:
            continue

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }   
            response = requests.get(current_url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            tags_to_search = ['p', 'article', 'section', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            for tag in tags_to_search:
                for element in soup.find_all(tag):
                    text = element.get_text().strip()
                    estimated_length = len(text) // 4  # Rough token estimate
                    if estimated_tokens + estimated_length <= token_limit:
                        collected_data.add(text)
                        estimated_tokens += estimated_length
                    else:
                        return ' '.join(collected_data)  # Return if limit reached

            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href)
                if is_internal_url(full_url, base_url):
                    urls_to_visit.add(full_url)

            visited_urls.add(current_url)
        except Exception as e:
            print(f"Error scraping {current_url}: {e}")

    return ' '.join(collected_data)

def extract_keywords(text):
    try:
        chat_completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.  Please analyze this text and provide a list of at least a hundred keywords in an array format.  Do not include any explanatory text; only respond with the array."
                },
                {
                    "role": "user",
                    "content": text,
                }
            ]
        )
        keywords = chat_completion['choices'][0]['message']['content']
        return keywords.split(',')
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return []

def generate_article_for_keyword(keyword):
    global last_request_time
    
    # Construct the initial messages for the chat, including setting the AI's role.
    messages = [
        {
            "role": "system",
            "content": "You are an AI trained to generate informative and original articles."
        },
        {
            "role": "user",
            "content": f"Write a detailed, informative, and original article about {keyword}."
        }
    ]
    
    while True:
        try:
            # Check if we need to rate limit our requests
            if last_request_time is not None:
                elapsed_time = time.time() - last_request_time
                if elapsed_time < min_time_between_requests:
                    time.sleep(min_time_between_requests - elapsed_time)
            
            # Make an API call to OpenAI's chat endpoint with the messages.
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use the chat model
                messages=messages
            )

            last_request_time = time.time()

            # The response will be in the last message of the chat response
            article = response['choices'][0]['message']['content']
            return article.strip()
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded. Waiting to retry...")
            time.sleep(60)  # Wait 60 seconds before retrying
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return ""

def generate_articles(keywords_array):
    articles = {}
    for keyword in keywords_array:
        article = generate_article_for_keyword(keyword)
        articles[keyword] = article  # Store the article with the keyword as the key
    return articles

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        scraped_content = scrape_url(url)
        keywords = extract_keywords(scraped_content)

        articles = generate_articles(keywords)
        first_keyword = keywords[0] if keywords else 'No keywords extracted'
        first_article = articles.get(first_keyword, 'No article generated')
        # return ', '.join(keywords)
        return first_article, articles
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)