import os
import time
import logging
import random
from dotenv import load_dotenv

from playwright.sync_api import sync_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)

class ChatGPTChater:
    def __init__(self):
        load_dotenv()
        self.base_url = "https://chatgpt.com/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }
        self.email = os.getenv('OPENAI_EMAIL')
        self.password = os.getenv('OPENAI_PASSWORD')
        self.logger = logging.getLogger(__name__)
        
    def _setup_browser(self, p):
        """Set up the browser with stealth mode and persistent context."""
        browser_args = [
            '--disable-blink-features=AutomationControlled', 
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-site-isolation-trials',
            '--disable-features=BlockInsecurePrivateNetworkRequests'
        ]
        
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36', 
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        ]

        # Create user data directory if it doesn't exist
        user_data_dir = "./chrome_data"
        if not os.path.exists(user_data_dir):
            os.makedirs(user_data_dir)

        browser = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=os.getenv('BROWSER_HEADLESS', 'true').lower() == 'true',
            args=browser_args,
            user_agent=random.choice(user_agents),
            viewport={'width': 1024, 'height': 768},
            ignore_https_errors=True,
            permissions=['geolocation']
        )
        
        return browser, browser.new_page()

    def _login(self, page):
        """Login to chatgpt.com using provided credentials, handling Cloudflare challenge."""
        try:
            page.goto("https://chatgpt.com/", timeout=60000)
            # Wait for potential Cloudflare challenge to resolve (e.g., network idle state)
            page.wait_for_load_state("networkidle", timeout=120000)
            # Wait for login button that indicates Cloudflare is bypassed; adjust selector if needed
            page.wait_for_selector('button:has-text("Log In")', timeout=120000)
            self.logger.info("Cloudflare bypassed, proceeding with login")
            
            # Proceed with login
            page.wait_for_selector('input[name="email"]', timeout=60000)
            page.fill('input[name="email"]', self.email)
            page.fill('input[name="password"]', self.password)
            with page.expect_navigation(timeout=60000):
                page.click('button[type="submit"]')
            page.wait_for_selector('div.chat-interface', timeout=60000)
            self.logger.info("Login successful on chatgpt.com")
        except Exception as e:
            self.logger.error(f"Login failed on chatgpt.com: {e}")
    
    def open_chat(self):
        with sync_playwright() as p:
            try:
                browser, page = self._setup_browser(p)
                self._login(page)
                page.wait_for_timeout(1000)
                time.sleep(10*60)
            except Exception as e:
                self.logger.error(f"Crawler failed: {e}")
                raise
            finally:
                if browser:
                    browser.close()

if __name__ == "__main__":
    scraper = ChatGPTChater()
    scraper.open_chat()
    # Example usage: uncomment to send a message
    # scraper.send_message("Hello ChatGPT, please help me with this query.")
    # scraper.close()
