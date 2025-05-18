## inference
### env
```bash
conda create -n maple-quiz python=3.9
conda activate maple-quiz
git clone https://github.com/JoonHyun814/mapleland-quiz.git
cd mapleland-quiz
pip install -r requirements.txt
```

### model download
https://drive.google.com/file/d/17FTxedEv6d-h7jOglB3HE9tQGV1dk3Ct/view?usp=sharing

이 모델 받아서 ckpt/test15.pt로 저장

### run inference
```bash
python inference/main.py
```
1. 모니터 선택 (0: 모니터 전체, 1: 메인 모니터, 2: 듀얼 모니터, 3 ...)
2. 영역 선택 후 s 키로 확정 (못바꿈 주의)
3. q로 프로그램 종료
