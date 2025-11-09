# Use the official Python 3.8 slim image as the base image
FROM python:3.10-slim

WORKDIR /var/www

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Define la variable de entorno FLASK_APP
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLASK_ENV=development

EXPOSE 5000

CMD ["flask", "run", "--reload"]