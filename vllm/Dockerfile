FROM quay.io/centos/centos:stream9 AS repos

FROM registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:ba060ec145e1b7ac3a3b25b89fc75f7d146746156990ff5858235eaa9d158e32

USER root
COPY --from=repos /etc/yum.repos.d/* /etc/yum.repos.d/
RUN echo '9-stream' > /etc/dnf/vars/stream && dnf config-manager --set-enabled crb && \
    dnf --nogpgcheck install -y gcc liburing-devel liburing strace

COPY main.c /tmp/uring-test.c
RUN gcc -O2 -Wall -Wextra \
    /tmp/uring-test.c \
    -o /usr/local/bin/uring-test \
    -luring && \
    rm /tmp/uring-test.c

RUN git clone https://github.com/aleskandro/InstantTensor.git && \
    cd InstantTensor && \
    ./checkout_submodules.sh && \
    pip install .

USER 1001