FROM python:3.10-slim

WORKDIR /app

# 필수 시스템 패키지 설치 (mysqlclient, mariadb 연동용)
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    python3-dev \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Python 패키지 설치
RUN pip install --no-cache-dir \
    flask \
    flask-sqlalchemy \
    mysqlclient \
    ultralytics \
    opencv-python \
    numpy \
    pillow \
    matplotlib \
    pymysql \
    python-dotenv \
    flask-wtf

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]
