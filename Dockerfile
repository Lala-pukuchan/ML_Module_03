FROM python:3.7

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install --upgrade pandas && \
    pip install --upgrade numpy && \
    pip install --upgrade matplotlib && \
    pip install --upgrade pycodestyle && \
    pip install --upgrade black && \
    pip install --upgrade scikit-learn

ENV PYTHONPATH /app

#ENTRYPOINT ["./test.sh"]
