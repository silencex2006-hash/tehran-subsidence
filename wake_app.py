import os
import sys
from playwright.sync_api import sync_playwright

def wake_app():
    url = os.environ.get("https://tehran-subsidence-hud5ns2vttfmo5gjdgwnfv.streamlit.app/")
    if not url:
        print("❌ متغیر APP_URL تنظیم نشده است.")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print(f"🌐 باز کردن آدرس: {url}")
        page.goto(url, wait_until="networkidle", timeout=60000)
        
        wake_button = page.get_by_text("get this app back up", exact=False)
        
        if wake_button.is_visible():
            print("💤 اپ در حالت خواب است. کلیک روی دکمه بیدارباش...")
            wake_button.click()
            page.wait_for_timeout(15000)
            print("✅ اپ با موفقیت بیدار شد.")
        else:
            print("☀️ اپ از قبل بیدار است.")
        
        browser.close()

if __name__ == "__main__":
    wake_app()
