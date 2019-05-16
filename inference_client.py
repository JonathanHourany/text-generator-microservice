"""gRPC client for text generating models"""

import grpc

from protos import gen_text_pb2
from protos import gen_text_pb2_grpc


def run(host, port):
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = gen_text_pb2_grpc.TextGeneratorStub(channel=channel)
        model_response = stub.GenerateText(gen_text_pb2.Text(payload='Luke!'))
        print(model_response)


if __name__ == "__main__":
    run('localhost', 5052)
