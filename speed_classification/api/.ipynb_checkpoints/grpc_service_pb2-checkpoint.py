# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grpc_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import api_pb2 as api__pb2
from . import request_status_pb2 as request__status__pb2
from . import server_status_pb2 as server__status__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='grpc_service.proto',
  package='nvidia.inferenceserver',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x12grpc_service.proto\x12\x16nvidia.inferenceserver\x1a\tapi.proto\x1a\x14request_status.proto\x1a\x13server_status.proto\"#\n\rStatusRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\"\x8c\x01\n\x0eStatusResponse\x12=\n\x0erequest_status\x18\x01 \x01(\x0b\x32%.nvidia.inferenceserver.RequestStatus\x12;\n\rserver_status\x18\x02 \x01(\x0b\x32$.nvidia.inferenceserver.ServerStatus\"\x1d\n\x0eProfileRequest\x12\x0b\n\x03\x63md\x18\x01 \x01(\t\"P\n\x0fProfileResponse\x12=\n\x0erequest_status\x18\x01 \x01(\x0b\x32%.nvidia.inferenceserver.RequestStatus\"\x1d\n\rHealthRequest\x12\x0c\n\x04mode\x18\x01 \x01(\t\"_\n\x0eHealthResponse\x12=\n\x0erequest_status\x18\x01 \x01(\x0b\x32%.nvidia.inferenceserver.RequestStatus\x12\x0e\n\x06health\x18\x02 \x01(\x08\"\x8b\x01\n\x0cInferRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x15\n\rmodel_version\x18\x02 \x01(\x03\x12=\n\tmeta_data\x18\x03 \x01(\x0b\x32*.nvidia.inferenceserver.InferRequestHeader\x12\x11\n\traw_input\x18\x04 \x03(\x0c\"\xa2\x01\n\rInferResponse\x12=\n\x0erequest_status\x18\x01 \x01(\x0b\x32%.nvidia.inferenceserver.RequestStatus\x12>\n\tmeta_data\x18\x02 \x01(\x0b\x32+.nvidia.inferenceserver.InferResponseHeader\x12\x12\n\nraw_output\x18\x03 \x03(\x0c\x32\xdb\x03\n\x0bGRPCService\x12Y\n\x06Status\x12%.nvidia.inferenceserver.StatusRequest\x1a&.nvidia.inferenceserver.StatusResponse\"\x00\x12\\\n\x07Profile\x12&.nvidia.inferenceserver.ProfileRequest\x1a\'.nvidia.inferenceserver.ProfileResponse\"\x00\x12Y\n\x06Health\x12%.nvidia.inferenceserver.HealthRequest\x1a&.nvidia.inferenceserver.HealthResponse\"\x00\x12V\n\x05Infer\x12$.nvidia.inferenceserver.InferRequest\x1a%.nvidia.inferenceserver.InferResponse\"\x00\x12`\n\x0bStreamInfer\x12$.nvidia.inferenceserver.InferRequest\x1a%.nvidia.inferenceserver.InferResponse\"\x00(\x01\x30\x01\x62\x06proto3')
  ,
  dependencies=[api__pb2.DESCRIPTOR,request__status__pb2.DESCRIPTOR,server__status__pb2.DESCRIPTOR,])




_STATUSREQUEST = _descriptor.Descriptor(
  name='StatusRequest',
  full_name='nvidia.inferenceserver.StatusRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_name', full_name='nvidia.inferenceserver.StatusRequest.model_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=100,
  serialized_end=135,
)


_STATUSRESPONSE = _descriptor.Descriptor(
  name='StatusResponse',
  full_name='nvidia.inferenceserver.StatusResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_status', full_name='nvidia.inferenceserver.StatusResponse.request_status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='server_status', full_name='nvidia.inferenceserver.StatusResponse.server_status', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=138,
  serialized_end=278,
)


_PROFILEREQUEST = _descriptor.Descriptor(
  name='ProfileRequest',
  full_name='nvidia.inferenceserver.ProfileRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cmd', full_name='nvidia.inferenceserver.ProfileRequest.cmd', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=280,
  serialized_end=309,
)


_PROFILERESPONSE = _descriptor.Descriptor(
  name='ProfileResponse',
  full_name='nvidia.inferenceserver.ProfileResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_status', full_name='nvidia.inferenceserver.ProfileResponse.request_status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=311,
  serialized_end=391,
)


_HEALTHREQUEST = _descriptor.Descriptor(
  name='HealthRequest',
  full_name='nvidia.inferenceserver.HealthRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mode', full_name='nvidia.inferenceserver.HealthRequest.mode', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=393,
  serialized_end=422,
)


_HEALTHRESPONSE = _descriptor.Descriptor(
  name='HealthResponse',
  full_name='nvidia.inferenceserver.HealthResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_status', full_name='nvidia.inferenceserver.HealthResponse.request_status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='health', full_name='nvidia.inferenceserver.HealthResponse.health', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=424,
  serialized_end=519,
)


_INFERREQUEST = _descriptor.Descriptor(
  name='InferRequest',
  full_name='nvidia.inferenceserver.InferRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_name', full_name='nvidia.inferenceserver.InferRequest.model_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_version', full_name='nvidia.inferenceserver.InferRequest.model_version', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meta_data', full_name='nvidia.inferenceserver.InferRequest.meta_data', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='raw_input', full_name='nvidia.inferenceserver.InferRequest.raw_input', index=3,
      number=4, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=522,
  serialized_end=661,
)


_INFERRESPONSE = _descriptor.Descriptor(
  name='InferResponse',
  full_name='nvidia.inferenceserver.InferResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_status', full_name='nvidia.inferenceserver.InferResponse.request_status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meta_data', full_name='nvidia.inferenceserver.InferResponse.meta_data', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='raw_output', full_name='nvidia.inferenceserver.InferResponse.raw_output', index=2,
      number=3, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=664,
  serialized_end=826,
)

_STATUSRESPONSE.fields_by_name['request_status'].message_type = request__status__pb2._REQUESTSTATUS
_STATUSRESPONSE.fields_by_name['server_status'].message_type = server__status__pb2._SERVERSTATUS
_PROFILERESPONSE.fields_by_name['request_status'].message_type = request__status__pb2._REQUESTSTATUS
_HEALTHRESPONSE.fields_by_name['request_status'].message_type = request__status__pb2._REQUESTSTATUS
_INFERREQUEST.fields_by_name['meta_data'].message_type = api__pb2._INFERREQUESTHEADER
_INFERRESPONSE.fields_by_name['request_status'].message_type = request__status__pb2._REQUESTSTATUS
_INFERRESPONSE.fields_by_name['meta_data'].message_type = api__pb2._INFERRESPONSEHEADER
DESCRIPTOR.message_types_by_name['StatusRequest'] = _STATUSREQUEST
DESCRIPTOR.message_types_by_name['StatusResponse'] = _STATUSRESPONSE
DESCRIPTOR.message_types_by_name['ProfileRequest'] = _PROFILEREQUEST
DESCRIPTOR.message_types_by_name['ProfileResponse'] = _PROFILERESPONSE
DESCRIPTOR.message_types_by_name['HealthRequest'] = _HEALTHREQUEST
DESCRIPTOR.message_types_by_name['HealthResponse'] = _HEALTHRESPONSE
DESCRIPTOR.message_types_by_name['InferRequest'] = _INFERREQUEST
DESCRIPTOR.message_types_by_name['InferResponse'] = _INFERRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StatusRequest = _reflection.GeneratedProtocolMessageType('StatusRequest', (_message.Message,), dict(
  DESCRIPTOR = _STATUSREQUEST,
  __module__ = 'grpc_service_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.StatusRequest)
  ))
_sym_db.RegisterMessage(StatusRequest)

StatusResponse = _reflection.GeneratedProtocolMessageType('StatusResponse', (_message.Message,), dict(
  DESCRIPTOR = _STATUSRESPONSE,
  __module__ = 'grpc_service_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.StatusResponse)
  ))
_sym_db.RegisterMessage(StatusResponse)

ProfileRequest = _reflection.GeneratedProtocolMessageType('ProfileRequest', (_message.Message,), dict(
  DESCRIPTOR = _PROFILEREQUEST,
  __module__ = 'grpc_service_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ProfileRequest)
  ))
_sym_db.RegisterMessage(ProfileRequest)

ProfileResponse = _reflection.GeneratedProtocolMessageType('ProfileResponse', (_message.Message,), dict(
  DESCRIPTOR = _PROFILERESPONSE,
  __module__ = 'grpc_service_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ProfileResponse)
  ))
_sym_db.RegisterMessage(ProfileResponse)

HealthRequest = _reflection.GeneratedProtocolMessageType('HealthRequest', (_message.Message,), dict(
  DESCRIPTOR = _HEALTHREQUEST,
  __module__ = 'grpc_service_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.HealthRequest)
  ))
_sym_db.RegisterMessage(HealthRequest)

HealthResponse = _reflection.GeneratedProtocolMessageType('HealthResponse', (_message.Message,), dict(
  DESCRIPTOR = _HEALTHRESPONSE,
  __module__ = 'grpc_service_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.HealthResponse)
  ))
_sym_db.RegisterMessage(HealthResponse)

InferRequest = _reflection.GeneratedProtocolMessageType('InferRequest', (_message.Message,), dict(
  DESCRIPTOR = _INFERREQUEST,
  __module__ = 'grpc_service_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferRequest)
  ))
_sym_db.RegisterMessage(InferRequest)

InferResponse = _reflection.GeneratedProtocolMessageType('InferResponse', (_message.Message,), dict(
  DESCRIPTOR = _INFERRESPONSE,
  __module__ = 'grpc_service_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferResponse)
  ))
_sym_db.RegisterMessage(InferResponse)



_GRPCSERVICE = _descriptor.ServiceDescriptor(
  name='GRPCService',
  full_name='nvidia.inferenceserver.GRPCService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=829,
  serialized_end=1304,
  methods=[
  _descriptor.MethodDescriptor(
    name='Status',
    full_name='nvidia.inferenceserver.GRPCService.Status',
    index=0,
    containing_service=None,
    input_type=_STATUSREQUEST,
    output_type=_STATUSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Profile',
    full_name='nvidia.inferenceserver.GRPCService.Profile',
    index=1,
    containing_service=None,
    input_type=_PROFILEREQUEST,
    output_type=_PROFILERESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Health',
    full_name='nvidia.inferenceserver.GRPCService.Health',
    index=2,
    containing_service=None,
    input_type=_HEALTHREQUEST,
    output_type=_HEALTHRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Infer',
    full_name='nvidia.inferenceserver.GRPCService.Infer',
    index=3,
    containing_service=None,
    input_type=_INFERREQUEST,
    output_type=_INFERRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='StreamInfer',
    full_name='nvidia.inferenceserver.GRPCService.StreamInfer',
    index=4,
    containing_service=None,
    input_type=_INFERREQUEST,
    output_type=_INFERRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_GRPCSERVICE)

DESCRIPTOR.services_by_name['GRPCService'] = _GRPCSERVICE

# @@protoc_insertion_point(module_scope)