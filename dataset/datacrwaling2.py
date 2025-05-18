import requests
from bs4 import BeautifulSoup
import os
import json

# 저장 폴더 생성
save_folder = 'mob_images'
os.makedirs(save_folder, exist_ok=True)

# 대상 URL
url = 'https://mapledb.kr/npc.php'
url = "https://mapledb.kr/mob.php"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

# 웹 페이지 요청
response = requests.get(url, headers=headers)
response.raise_for_status()

# HTML 파싱
soup = BeautifulSoup(response.text, 'html.parser')

# 모든 NPC 항목 선택
npc_list = soup.select('div.main-content > div > a')

with open("inference/database_idx.json",encoding="UTF-8") as f:
    database_idx = json.load(f)

gt_npc_list = list(database_idx.values())

for npc in npc_list:
    img_tag = npc.select_one('div > img')
    name_tag = npc.select_one('div > h3')

    if not img_tag or not name_tag:
        continue  # 요소가 없으면 건너뜀

    img_url = img_tag['src']
    npc_name = name_tag.text.strip()
    filename = os.path.join(save_folder, f"{npc_name}.png")

    if npc_name in gt_npc_list:
        # 이미지 다운로드
        img_response = requests.get(img_url, headers=headers)
        if img_response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(img_response.content)
            print(f"[✔] 저장 완료: {filename}")
        else:
            print(f"[✘] 이미지 다운로드 실패: {npc_name}")
    elif npc_name.replace(" ","") in gt_npc_list:
        # 이미지 다운로드
        filename = os.path.join(save_folder, f"{npc_name.replace(' ','')}.png")
        img_response = requests.get(img_url, headers=headers)
        if img_response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(img_response.content)
            print(f"[✔] 저장 완료: {filename}")
        else:
            print(f"[✘] 이미지 다운로드 실패: {npc_name}")