FROM python:3.12
WORKDIR /src
COPY . /src
RUN pip install -r requirements.txt && apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx
CMD ["python", "main.py"]