FROM python:3.7-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get -y --no-install-recommends install libgomp1
RUN pip install --trusted-host pypi.python.org -r requirements.txt
     
EXPOSE 8080
ENV PORT 8080

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
