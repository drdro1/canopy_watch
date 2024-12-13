FROM python:3.10.6-buster
COPY canopywatch canopywatch
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY canopy-442811-d3f55a231cd0.json canopy-442811-d3f55a231cd0.json
RUN pip install --upgrade pip
RUN pip install .
CMD uvicorn canopywatch.api.fast:app --host 0.0.0.0 --port $PORT
