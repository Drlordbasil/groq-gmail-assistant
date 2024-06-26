import time
import random
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import trafilatura
from selenium.common.exceptions import WebDriverException, NoSuchElementException, TimeoutException
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

max_content_length = 1000

class SelectorRL:
    def __init__(self):
        self.selectors = {
            "google": {
                "search_box": ["input[name='q']", "textarea[name='q']", "#search-input"],
                "result": ["div.g", "div.tF2Cxc", "div.yuRUbf"]
            },
            "bing": {
                "search_box": ["input[name='q']", "#sb_form_q"],
                "result": ["li.b_algo", "div.b_title", "h2"]
            },
            "brave": {
                "search_box": ["input[name='q']", "#searchbox"],
                "result": ["div.snippet", "div.fdb", "div.result"]
            }
        }
        self.q_values = {engine: {selector: 0 for selector_type in selectors.values() for selector in selector_type} for engine, selectors in self.selectors.items()}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def get_selector(self, engine, selector_type):
        if random.random() < self.epsilon:
            return random.choice(self.selectors[engine][selector_type])
        else:
            return max(self.selectors[engine][selector_type], key=lambda s: self.q_values[engine][s])

    def update_q_value(self, engine, selector, reward):
        self.q_values[engine][selector] += self.learning_rate * (reward - self.q_values[engine][selector])

    def add_new_selector(self, engine, selector_type, new_selector):
        if new_selector not in self.selectors[engine][selector_type]:
            self.selectors[engine][selector_type].append(new_selector)
            self.q_values[engine][new_selector] = 0

    def save_state(self, filename='selector_rl_state.json'):
        state = {
            'q_values': self.q_values,
            'selectors': self.selectors
        }
        with open(filename, 'w') as f:
            json.dump(state, f)

    def load_state(self, filename='selector_rl_state.json'):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            self.q_values = state['q_values']
            self.selectors = state['selectors']
        except FileNotFoundError:
            print("No saved state found. Starting with default values.")

class WebResearchTool:
    def __init__(self):
        self.max_content_length = 2000
        self.selector_rl = SelectorRL()
        self.selector_rl.load_state()
        self.vectorizer = TfidfVectorizer()

    def _initialize_webdriver(self):
        options = webdriver.ChromeOptions()
        service = ChromeService(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def find_new_selector(self, driver, element_type):
        if element_type == "search_box":
            potential_selectors = driver.find_elements(By.XPATH, "//input[@type='text'] | //input[@type='search'] | //textarea")
        else:  # result
            potential_selectors = driver.find_elements(By.XPATH, "//div[.//a] | //li[.//a] | //h2[.//a]")

        for element in potential_selectors:
            try:
                selector = self.get_css_selector(driver, element)
                return selector
            except:
                continue
        return None

    def get_css_selector(self, driver, element):
        return driver.execute_script("""
            var path = [];
            var element = arguments[0];
            while (element.nodeType === Node.ELEMENT_NODE) {
                var selector = element.nodeName.toLowerCase();
                if (element.id) {
                    selector += '#' + element.id;
                    path.unshift(selector);
                    break;
                } else {
                    var sibling = element;
                    var nth = 1;
                    while (sibling.previousElementSibling) {
                        sibling = sibling.previousElementSibling;
                        if (sibling.nodeName.toLowerCase() == selector)
                            nth++;
                    }
                    if (nth != 1)
                        selector += ":nth-of-type("+nth+")";
                }
                path.unshift(selector);
                element = element.parentNode;
            }
            return path.join(' > ');
        """, element)

    def extract_text_from_url(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
            if text and len(text) >= 50:
                return text

            driver = self._initialize_webdriver()
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
                element.decompose()
            text = ' '.join(p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 20)
            return text if len(text) >= 50 else None
        except Exception as e:
            print(f"Error extracting text from URL {url}: {e}")
            return None
        finally:
            if 'driver' in locals():
                driver.quit()

    def crawl_website(self, url, max_pages=5, progress_callback: callable = None):
        visited = set()
        to_visit = [url]
        graph = nx.DiGraph()
        content = {}

        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue

            visited.add(current_url)
            if progress_callback:
                progress_callback(f"Crawling: {current_url}")

            try:
                response = requests.get(current_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract content
                text = self.extract_text_from_url(current_url)
                if text:
                    content[current_url] = text

                # Find links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = requests.compat.urljoin(current_url, href)
                    if full_url.startswith(url):  # Stay on the same domain
                        graph.add_edge(current_url, full_url)
                        if full_url not in visited:
                            to_visit.append(full_url)

            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error crawling {current_url}: {e}")

        return graph, content

    def calculate_similarity(self, query, text):
        tfidf_matrix = self.vectorizer.fit_transform([query, text])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def web_research(self, query, progress_callback: callable = None):
        if progress_callback:
            progress_callback("Starting web research...")

        combined_query = query
        search_engines = [
            ("https://www.google.com/search", "google"),
            ("https://www.bing.com/search", "bing"),
            ("https://search.brave.com/search", "brave")
        ]
        search_results = []

        for engine_url, engine_name in search_engines:
            if progress_callback:
                progress_callback(f"Using search engine: {engine_name}")

            try:
                driver = self._initialize_webdriver()
                driver.get(engine_url)
                
                search_box_selector = self.selector_rl.get_selector(engine_name, "search_box")
                result_selector = self.selector_rl.get_selector(engine_name, "result")
                
                try:
                    search_box = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, search_box_selector))
                    )
                    search_box.send_keys(combined_query)
                    search_box.send_keys(Keys.RETURN)
                    
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, result_selector))
                    )
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    results = soup.select(result_selector)[:5]  # Top 5 results
                    
                    if results:
                        self.selector_rl.update_q_value(engine_name, search_box_selector, 1)
                        self.selector_rl.update_q_value(engine_name, result_selector, 1)
                    else:
                        if progress_callback:
                            progress_callback(f"No results found with current selectors for {engine_name}. Attempting to find new selectors.")
                        new_search_box_selector = self.find_new_selector(driver, "search_box")
                        new_result_selector = self.find_new_selector(driver, "result")
                        
                        if new_search_box_selector and new_result_selector:
                            self.selector_rl.add_new_selector(engine_name, "search_box", new_search_box_selector)
                            self.selector_rl.add_new_selector(engine_name, "result", new_result_selector)
                            if progress_callback:
                                progress_callback(f"New selectors found for {engine_name}: Search box: {new_search_box_selector}, Result: {new_result_selector}")
                            
                            # Retry with new selectors
                            driver.get(engine_url)
                            search_box = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, new_search_box_selector))
                            )
                            search_box.send_keys(combined_query)
                            search_box.send_keys(Keys.RETURN)
                            
                            WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, new_result_selector))
                            )
                            
                            soup = BeautifulSoup(driver.page_source, 'html.parser')
                            results = soup.select(new_result_selector)[:5]
                    
                    for result in results:
                        link = result.select_one('a')
                        if link and link.get('href'):
                            url = link['href']
                            if url.startswith('http'):
                                graph, crawled_content = self.crawl_website(url, progress_callback=progress_callback)
                                for page_url, content in crawled_content.items():
                                    similarity = self.calculate_similarity(combined_query, content)
                                    if similarity > 0.1:  # Adjust threshold as needed
                                        search_results.append({
                                            "title": link.get_text(),
                                            "link": page_url,
                                            "content": content,
                                            "similarity": similarity
                                        })
                except (NoSuchElementException, TimeoutException) as e:
                    if progress_callback:
                        progress_callback(f"Error interacting with search engine {engine_url}: {e}")
                    self.selector_rl.update_q_value(engine_name, search_box_selector, -1)
                    self.selector_rl.update_q_value(engine_name, result_selector, -1)
            except WebDriverException as e:
                if progress_callback:
                    progress_callback(f"WebDriver error with search engine {engine_url}: {e}")
            finally:
                driver.quit()

        self.selector_rl.save_state()

        if not search_results:
            return f"No results found for the query: {combined_query}"

        # Sort results by similarity
        search_results.sort(key=lambda x: x['similarity'], reverse=True)

        aggregated_content = ""
        for result in search_results:
            if len(aggregated_content) + len(result['content']) <= self.max_content_length:
                aggregated_content += f"[Source: {result['link']}]\n{result['content']}\n\n"
            else:
                remaining_chars = self.max_content_length - len(aggregated_content)
                aggregated_content += f"[Source: {result['link']}]\n{result['content'][:remaining_chars]}"
                break

        if progress_callback:
            progress_callback("Web research completed.")

        return aggregated_content.strip() if aggregated_content else f"Unable to retrieve relevant content for the query: {combined_query}"

# Example usage
if __name__ == "__main__":
    research_tool = WebResearchTool()
    user_prompt = "What are the latest advancements in AI?"
    assistant_query = "Focus on breakthroughs in natural language processing and computer vision"

    def progress_update(message):
        print(f"Progress: {message}")

    results = research_tool.web_research(query=f"{user_prompt} {assistant_query}", progress_callback=progress_update)
    print(f"Research results:\n\n{results}")
