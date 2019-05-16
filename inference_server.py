"""gRPC server for text generating models"""

from concurrent import futures
import time
import math
import logging
from argparse import ArgumentParser
import grpc

import inference
from protos import gen_text_pb2
from protos import gen_text_pb2_grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class TextGeneratorServicer(gen_text_pb2_grpc.TextGeneratorServicer):
    """Provides methods that implement functionality of TextGeneratorServicer.

    This is a singleton. Loading a Tensorflow model can be a slow process. To avoid the unnessarry overhead, the
    TF model is cached as a class variable after the first time it's loaded
    """

    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            model_weights_path = "model_weights/"
            cls.model = inference.load_model_from_weights(model_weights_path=model_weights_path)
        return cls.model

    def GenerateText(self, request, context):
        model = self.__class__.get_model()
        model_response = inference.generate_text(model, request.payload, text2embedd=inference.CHAR2EMBEDD_MAP,
                                                 embedd2text=inference.EMBEDD2CHAR_MAP)

        return gen_text_pb2.Text(payload=model_response)


def serve(port, max_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    gen_text_pb2_grpc.add_TextGeneratorServicer_to_server(servicer=TextGeneratorServicer(), server=server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print(f"Server online. Listing on port {args.port}")

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":

    parser = ArgumentParser(description='Generate text from neural network')
    parser.add_argument('--port', type=int, help='port number', required=False, default=5052)
    parser.add_argument('--max_workers', type=int, help='# max workers', required=False, default=2)
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.max_workers)
