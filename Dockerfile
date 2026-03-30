
FROM python:3.10-slim


WORKDIR /app


RUN pip install --no-cache-dir tensorflow numpy pillow scikit-image scipy


CMD ["python", "lab_2.py"]