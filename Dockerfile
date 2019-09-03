FROM python:3.7.4-buster

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt
