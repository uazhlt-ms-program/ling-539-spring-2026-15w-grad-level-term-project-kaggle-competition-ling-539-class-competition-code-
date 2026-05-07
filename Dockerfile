FROM python:3.11-slim

LABEL description="Reproducible environment for the LING 539 Kaggle text classifier."

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["/bin/bash"]