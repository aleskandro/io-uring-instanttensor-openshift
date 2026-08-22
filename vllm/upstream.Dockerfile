FROM vllm/vllm-openai:latest
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/aleskandro/InstantTensor.git && \
    cd InstantTensor && \
    ./checkout_submodules.sh && \
    pip install .

USER 1001