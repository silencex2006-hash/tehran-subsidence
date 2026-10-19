import requests
import time
import sys

# ⚠️ اینجا آدرس اپلیکیشن استریم‌لیت خودتان را جایگزین کنید (حتماً با https)
URL = "https://tehran-subsidence-hud5ns2vttfmo5gjdgwnfv.streamlit.app/" 

def check_app():
    # هدر مرورگر برای اینکه سرور Streamlit درخواست را بلاک نکند
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # تلاش ۳ باره برای جلوگیری از خطای الکی
    for attempt in range(3):
        try:
            print(f"Attempt {attempt + 1}: Pinging {URL}...")
            response = requests.get(URL, headers=headers, timeout=60)
            
            # اگر کد 200 دریافت کردیم، یعنی اپ آنلاین است
            if response.status_code == 200:
                print("✅ Success: App is online and responding.")
                return True
            else:
                print(f"⚠️ App returned status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
        
        # ۱۵ ثانیه صبر و تلاش مجدد
        time.sleep(15) 
        
    print("🚨 Failed to reach the app after 3 attempts.")
    return False

if __name__ == "__main__":
    if not check_app():
        sys.exit(1)  # اگر اپ واقعاً خراب بود، ارور می‌دهد
    else:
        sys.exit(0)  # اگر موفق بود، خارج می‌شود (تیک سبز)
