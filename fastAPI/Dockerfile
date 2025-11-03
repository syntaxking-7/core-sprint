FROM python:3.11

WORKDIR /app
COPY . /app
RUN pip install pandas
RUN pip install asyncpg
RUN pip install -r requirements.txt
WORKDIR /app/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port=8080"]
