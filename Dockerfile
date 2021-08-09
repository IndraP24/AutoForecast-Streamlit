FROM python:3.8

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

COPY . .

CMD streamlit run --server.port $PORT app.py