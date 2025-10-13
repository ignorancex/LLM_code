import os
from playwright.sync_api import sync_playwright
from PIL import Image
from io import BytesIO

def take_screenshot_from_url(url, output_file = "screenshot.png", do_it_again=False):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    if output_file is not None and os.path.exists(output_file) and not do_it_again:
        print(f"{output_file} exists!")
        return

    screenshot_image = None

    try:
        with sync_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)
            screenshot_bytes = page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)

            if output_file is None:
                screenshot_image = Image.open(BytesIO(screenshot_bytes))

            browser.close()
            
    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        screenshot_image = Image.new('RGB', (1280, 960), color='white')
        if output_file:
            screenshot_image.save(output_file)
    finally:
        return screenshot_image



def take_screenshot_from_html_content(page, html_content: str, output_file: str = "screenshot.png", do_it_again=False) -> Image:
    if output_file is not None and os.path.exists(output_file) and not do_it_again:
        print(f"{output_file} exists!")
        return


    screenshot_image = None
    try:
        # Choose a browser, e.g., Chromium, Firefox, or WebKit

        # Set the HTML content of the page
        page.set_content(html_content, timeout=60000)

        # Take the screenshot
        screenshot_bytes = page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)

        if output_file is None:
            screenshot_image = Image.open(BytesIO(screenshot_bytes))

        
    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        screenshot_image = Image.new('RGB', (1280, 960), color='white')
        if output_file:
            screenshot_image.save(output_file)
    finally:
        return screenshot_image



if __name__ == "__main__":
    url = "https://www.example.com"
    image = take_screenshot_from_url(url)

    # Show the image
    if image:
        image.show()
