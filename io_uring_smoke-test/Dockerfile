FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        liburing-dev && \
    rm -rf /var/lib/apt/lists/*

COPY main.c /tmp/uring-test.c

RUN gcc -O2 -Wall -Wextra \
    /tmp/uring-test.c \
    -o /usr/local/bin/uring-test \
    -luring && \
    rm /tmp/uring-test.c

ENTRYPOINT ["/usr/local/bin/uring-test"]
