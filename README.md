# YOLOv8 기반 히트맵 분석 시스템

동영상에서 YOLOv8m 모델을 이용하여 사람을 감지하고, 인구 밀집도 히트맵을 생성하는 웹 애플리케이션입니다.

## 주요 기능

- 동영상 업로드 및 분석
- 히트맵 생성 및 시각화
- 시간대별 통계 분석
- 이동 경로 분석
- 객체 탐지 결과 영상 생성

## 기술 스택

- **백엔드**: Flask
- **데이터베이스**: MariaDB
- **객체 탐지**: YOLOv8m
- **영상 처리**: OpenCV
- **데이터 시각화**: Chart.js

## 설치 방법

### 필수 조건

- Python 3.8 이상
- MariaDB
- FFmpeg

### 설치 과정

1. 저장소 복제
   ```bash
   git clone [repository-url]
   cd heatmap-flask
   ```

2. 가상 환경 설정
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

4. MariaDB 설정
   ```sql
   CREATE DATABASE heatmap_db;
   CREATE USER 'admin'@'localhost' IDENTIFIED BY 'admin';
   GRANT ALL PRIVILEGES ON heatmap_db.* TO 'admin'@'localhost';
   FLUSH PRIVILEGES;
   ```

5. 환경 변수 설정
   ```bash
   # .env 파일 생성
   SECRET_KEY=your_secret_key
   DB_HOST=localhost
   DB_USER=admin
   DB_PASS=admin
   DB_NAME=heatmap_db
   ```

6. 실행
   ```bash
   python app.py
   ```

## 프로젝트 구조

```
heatmap-flask/
├── app.py               # 메인 애플리케이션
├── models/             
│   ├── database.py     # DB 연결
│   ├── video_analysis.py # 분석 모델
│   └── yolo_detector.py # YOLOv8 감지
├── routes/             
│   ├── upload_routes.py # 업로드 처리
│   └── heatmap_routes.py # 히트맵 처리
├── static/             # 정적 파일
├── templates/          # 템플릿
├── uploads/            # 업로드 파일
└── processed/          # 처리된 파일
```

## 참고사항

- 동영상은 720p로 자동 변환되어 처리됩니다.
- 히트맵은 10프레임 단위로 생성됩니다.
- YOLOv8m 모델은 첫 실행 시 자동으로 다운로드됩니다.

## 라이선스

MIT License 