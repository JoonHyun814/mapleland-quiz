from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import requests
import os

options = Options()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://maplequiz.com/")
time.sleep(3)  # JS 로딩 대기

landname_list = ["aquarium","ludibrium","alien","under-city","leefre","moorng","white-city"]

for landname in landname_list:
    # #maple-island 내부의 문제 리스트 가져오기
    dir_name = landname
    container = driver.find_element(By.ID, landname)

    for i in range(1,3):
        items = container.find_elements(By.CSS_SELECTOR, f"div > div:nth-child({i}) > div > div")

        # 이미지 저장용 폴더
        os.makedirs(f"maple_images/{dir_name}", exist_ok=True)

        for i, item in enumerate(items, start=1):
            try:
                # 이미지 요소 찾기
                img_element = item.find_element(By.CSS_SELECTOR, "div > div > img")
                img_src = img_element.get_attribute("src")

                # 텍스트 요소 찾기
                text_element = item.find_element(By.CSS_SELECTOR, "span")
                text_content = text_element.text.strip()

                print(f"문제 {i}: 텍스트 = {text_content}")
                print(f"문제 {i}: 이미지 URL = {img_src}")

                # 이미지 다운로드
                response = requests.get(img_src)
                if response.status_code == 200:
                    # 파일명에 텍스트나 번호 넣기(특수문자 제거 필수)
                    safe_text = "".join(c for c in text_content if c.isalnum() or c in (' ', '_')).rstrip()
                    filename = f"maple_images/{dir_name}/{i}_{safe_text[:20]}.png"

                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"✅ 이미지 저장: {filename}")
                else:
                    print(f"❌ 이미지 다운로드 실패 (상태코드 {response.status_code})")
            except Exception as e:
                print(f"⚠️ 문제 {i} 처리 중 오류 발생: {e}")

driver.quit()
