FROM python:3.13-slim
RUN apt update && apt install -y x11-utils
WORKDIR /src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DISPLAY=host.docker.internal:0.0
CMD ["python3", "src/main.py"]