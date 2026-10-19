import asyncio
import sys
from playwright.async_api import async_playwright

URL = "https://tehran-subsidence-hud5ns2vttfmo5gjdgwnfv.streamlit.app/"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print(f"Opening {URL}...")
            await page.goto(URL, wait_until="domcontentloaded", timeout=120000)
            await page.wait_for_timeout(5000) 
            
            wake_btn = page.get_by_role("button", name="Yes, get this app back up!")
            
            if await wake_btn.count() > 0:
                print("App is sleeping. Clicking wake up button...")
                await wake_btn.click()
                await page.wait_for_timeout(60000) 
                print("✅ App wake up signal sent successfully.")
            else:
                print("✅ App is already awake. No action needed.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            raise 
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
