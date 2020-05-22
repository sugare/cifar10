# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: request_status.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='request_status.proto',
  package='nvidia.inferenceserver',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x14request_status.proto\x12\x16nvidia.inferenceserver\"|\n\rRequestStatus\x12\x37\n\x04\x63ode\x18\x01 \x01(\x0e\x32).nvidia.inferenceserver.RequestStatusCode\x12\x0b\n\x03msg\x18\x02 \x01(\t\x12\x11\n\tserver_id\x18\x03 \x01(\t\x12\x12\n\nrequest_id\x18\x04 \x01(\x04*\x9e\x01\n\x11RequestStatusCode\x12\x0b\n\x07INVALID\x10\x00\x12\x0b\n\x07SUCCESS\x10\x01\x12\x0b\n\x07UNKNOWN\x10\x02\x12\x0c\n\x08INTERNAL\x10\x03\x12\r\n\tNOT_FOUND\x10\x04\x12\x0f\n\x0bINVALID_ARG\x10\x05\x12\x0f\n\x0bUNAVAILABLE\x10\x06\x12\x0f\n\x0bUNSUPPORTED\x10\x07\x12\x12\n\x0e\x41LREADY_EXISTS\x10\x08\x62\x06proto3')
)

_REQUESTSTATUSCODE = _descriptor.EnumDescriptor(
  name='RequestStatusCode',
  full_name='nvidia.inferenceserver.RequestStatusCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='INVALID', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INTERNAL', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOT_FOUND', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INVALID_ARG', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNAVAILABLE', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNSUPPORTED', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALREADY_EXISTS', index=8, number=8,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=175,
  serialized_end=333,
)
_sym_db.RegisterEnumDescriptor(_REQUESTSTATUSCODE)

RequestStatusCode = enum_type_wrapper.EnumTypeWrapper(_REQUESTSTATUSCODE)
INVALID = 0
SUCCESS = 1
UNKNOWN = 2
INTERNAL = 3
NOT_FOUND = 4
INVALID_ARG = 5
UNAVAILABLE = 6
UNSUPPORTED = 7
ALREADY_EXISTS = 8



_REQUESTSTATUS = _descriptor.Descriptor(
  name='RequestStatus',
  full_name='nvidia.inferenceserver.RequestStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='nvidia.inferenceserver.RequestStatus.code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='msg', full_name='nvidia.inferenceserver.RequestStatus.msg', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='server_id', full_name='nvidia.inferenceserver.RequestStatus.server_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='request_id', full_name='nvidia.inferenceserver.RequestStatus.request_id', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=48,
  serialized_end=172,
)

_REQUESTSTATUS.fields_by_name['code'].enum_type = _REQUESTSTATUSCODE
DESCRIPTOR.message_types_by_name['RequestStatus'] = _REQUESTSTATUS
DESCRIPTOR.enum_types_by_name['RequestStatusCode'] = _REQUESTSTATUSCODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RequestStatus = _reflection.GeneratedProtocolMessageType('RequestStatus', (_message.Message,), dict(
  DESCRIPTOR = _REQUESTSTATUS,
  __module__ = 'request_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.RequestStatus)
  ))
_sym_db.RegisterMessage(RequestStatus)


# @@protoc_insertion_point(module_scope)
