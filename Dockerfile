FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    libsqlite3-dev \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치 및 protobuf 다운그레이드
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir protobuf==3.20.3

# 디렉토리 생성
RUN mkdir -p DB && \
    chmod -R 777 DB

# Supervisor 설정 파일 생성
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 코드 복사
COPY . .

# 환경 변수
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# Supervisor로 프로세스 실행
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
