# 이동평균선 백테스팅 분석기

이 프로젝트는 한국 주식의 이동평균선 전략을 백테스팅하고, 최적의 이동평균선 및 전략의 안정성을 분석하는 Streamlit 기반 웹앱입니다.

## 주요 기능
- **종목 티커 입력**: 분석할 주식의 종목 코드 입력 (예: 삼성전자 - 005930)
- **매도 수수료 설정**: 매도 시 발생하는 수수료 및 세금 입력
- **분석 기간 설정**: 백테스팅 시작일과 종료일 지정
- **이동평균선 최적화**: 지정된 기간에서 가장 수익성이 좋은 이동평균선 탐색
- **종합 안정성 분석**: 다양한 시장 상황에서 전략의 안정성 평가
- **성과 지표 및 차트 시각화**: 수익률, 샤프비율, 최대낙폭, 매매빈도 등

## 설치 및 실행 방법
1. Python 3.11+ 환경 준비
2. 프로젝트 폴더에서 아래 명령어 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run "Korean Stock Moving Average Analysis.py"
```

## requirements.txt 예시
```
streamlit
pandas
numpy
finance-datareader
plotly
```

## Streamlit Community Cloud 배포
- GitHub 저장소에 업로드 후 https://streamlit.io/cloud 에서 바로 배포 가능

## 라이선스
MIT License
