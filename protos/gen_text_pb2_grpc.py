# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from protos import gen_text_pb2 as protos_dot_gen__text__pb2


class TextGeneratorStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.SendTextSeed = channel.unary_unary(
        '/TextGenerator/SendTextSeed',
        request_serializer=protos_dot_gen__text__pb2.Text.SerializeToString,
        response_deserializer=protos_dot_gen__text__pb2.Text.FromString,
        )
    self.GenerateText = channel.unary_unary(
        '/TextGenerator/GenerateText',
        request_serializer=protos_dot_gen__text__pb2.Text.SerializeToString,
        response_deserializer=protos_dot_gen__text__pb2.Text.FromString,
        )


class TextGeneratorServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def SendTextSeed(self, request, context):
    """Sends the starting string to the model
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GenerateText(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_TextGeneratorServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'SendTextSeed': grpc.unary_unary_rpc_method_handler(
          servicer.SendTextSeed,
          request_deserializer=protos_dot_gen__text__pb2.Text.FromString,
          response_serializer=protos_dot_gen__text__pb2.Text.SerializeToString,
      ),
      'GenerateText': grpc.unary_unary_rpc_method_handler(
          servicer.GenerateText,
          request_deserializer=protos_dot_gen__text__pb2.Text.FromString,
          response_serializer=protos_dot_gen__text__pb2.Text.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'TextGenerator', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
