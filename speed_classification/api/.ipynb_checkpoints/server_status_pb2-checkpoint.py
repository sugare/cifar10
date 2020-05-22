# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: server_status.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from . import model_config_pb2 as model__config__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='server_status.proto',
  package='nvidia.inferenceserver',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x13server_status.proto\x12\x16nvidia.inferenceserver\x1a\x12model_config.proto\"4\n\x0cStatDuration\x12\r\n\x05\x63ount\x18\x01 \x01(\x04\x12\x15\n\rtotal_time_ns\x18\x02 \x01(\x04\"K\n\x12StatusRequestStats\x12\x35\n\x07success\x18\x01 \x01(\x0b\x32$.nvidia.inferenceserver.StatDuration\"L\n\x13ProfileRequestStats\x12\x35\n\x07success\x18\x01 \x01(\x0b\x32$.nvidia.inferenceserver.StatDuration\"K\n\x12HealthRequestStats\x12\x35\n\x07success\x18\x01 \x01(\x0b\x32$.nvidia.inferenceserver.StatDuration\"\xec\x01\n\x11InferRequestStats\x12\x35\n\x07success\x18\x01 \x01(\x0b\x32$.nvidia.inferenceserver.StatDuration\x12\x34\n\x06\x66\x61iled\x18\x02 \x01(\x0b\x32$.nvidia.inferenceserver.StatDuration\x12\x35\n\x07\x63ompute\x18\x03 \x01(\x0b\x32$.nvidia.inferenceserver.StatDuration\x12\x33\n\x05queue\x18\x04 \x01(\x0b\x32$.nvidia.inferenceserver.StatDuration\"\xbf\x02\n\x12ModelVersionStatus\x12<\n\x0bready_state\x18\x01 \x01(\x0e\x32\'.nvidia.inferenceserver.ModelReadyState\x12O\n\x0binfer_stats\x18\x02 \x03(\x0b\x32:.nvidia.inferenceserver.ModelVersionStatus.InferStatsEntry\x12\x1d\n\x15model_execution_count\x18\x03 \x01(\x04\x12\x1d\n\x15model_inference_count\x18\x04 \x01(\x04\x1a\\\n\x0fInferStatsEntry\x12\x0b\n\x03key\x18\x01 \x01(\r\x12\x38\n\x05value\x18\x02 \x01(\x0b\x32).nvidia.inferenceserver.InferRequestStats:\x02\x38\x01\"\xf4\x01\n\x0bModelStatus\x12\x33\n\x06\x63onfig\x18\x01 \x01(\x0b\x32#.nvidia.inferenceserver.ModelConfig\x12N\n\x0eversion_status\x18\x02 \x03(\x0b\x32\x36.nvidia.inferenceserver.ModelStatus.VersionStatusEntry\x1a`\n\x12VersionStatusEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\x39\n\x05value\x18\x02 \x01(\x0b\x32*.nvidia.inferenceserver.ModelVersionStatus:\x02\x38\x01\"\xeb\x03\n\x0cServerStatus\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12=\n\x0bready_state\x18\x07 \x01(\x0e\x32(.nvidia.inferenceserver.ServerReadyState\x12\x11\n\tuptime_ns\x18\x03 \x01(\x04\x12K\n\x0cmodel_status\x18\x04 \x03(\x0b\x32\x35.nvidia.inferenceserver.ServerStatus.ModelStatusEntry\x12@\n\x0cstatus_stats\x18\x05 \x01(\x0b\x32*.nvidia.inferenceserver.StatusRequestStats\x12\x42\n\rprofile_stats\x18\x06 \x01(\x0b\x32+.nvidia.inferenceserver.ProfileRequestStats\x12@\n\x0chealth_stats\x18\x08 \x01(\x0b\x32*.nvidia.inferenceserver.HealthRequestStats\x1aW\n\x10ModelStatusEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x32\n\x05value\x18\x02 \x01(\x0b\x32#.nvidia.inferenceserver.ModelStatus:\x02\x38\x01*t\n\x0fModelReadyState\x12\x11\n\rMODEL_UNKNOWN\x10\x00\x12\x0f\n\x0bMODEL_READY\x10\x01\x12\x15\n\x11MODEL_UNAVAILABLE\x10\x02\x12\x11\n\rMODEL_LOADING\x10\x03\x12\x13\n\x0fMODEL_UNLOADING\x10\x04*\x86\x01\n\x10ServerReadyState\x12\x12\n\x0eSERVER_INVALID\x10\x00\x12\x17\n\x13SERVER_INITIALIZING\x10\x01\x12\x10\n\x0cSERVER_READY\x10\x02\x12\x12\n\x0eSERVER_EXITING\x10\x03\x12\x1f\n\x1bSERVER_FAILED_TO_INITIALIZE\x10\nb\x06proto3')
  ,
  dependencies=[model__config__pb2.DESCRIPTOR,])

_MODELREADYSTATE = _descriptor.EnumDescriptor(
  name='ModelReadyState',
  full_name='nvidia.inferenceserver.ModelReadyState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MODEL_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MODEL_READY', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MODEL_UNAVAILABLE', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MODEL_LOADING', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MODEL_UNLOADING', index=4, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1655,
  serialized_end=1771,
)
_sym_db.RegisterEnumDescriptor(_MODELREADYSTATE)

ModelReadyState = enum_type_wrapper.EnumTypeWrapper(_MODELREADYSTATE)
_SERVERREADYSTATE = _descriptor.EnumDescriptor(
  name='ServerReadyState',
  full_name='nvidia.inferenceserver.ServerReadyState',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SERVER_INVALID', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SERVER_INITIALIZING', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SERVER_READY', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SERVER_EXITING', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SERVER_FAILED_TO_INITIALIZE', index=4, number=10,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1774,
  serialized_end=1908,
)
_sym_db.RegisterEnumDescriptor(_SERVERREADYSTATE)

ServerReadyState = enum_type_wrapper.EnumTypeWrapper(_SERVERREADYSTATE)
MODEL_UNKNOWN = 0
MODEL_READY = 1
MODEL_UNAVAILABLE = 2
MODEL_LOADING = 3
MODEL_UNLOADING = 4
SERVER_INVALID = 0
SERVER_INITIALIZING = 1
SERVER_READY = 2
SERVER_EXITING = 3
SERVER_FAILED_TO_INITIALIZE = 10



_STATDURATION = _descriptor.Descriptor(
  name='StatDuration',
  full_name='nvidia.inferenceserver.StatDuration',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='count', full_name='nvidia.inferenceserver.StatDuration.count', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='total_time_ns', full_name='nvidia.inferenceserver.StatDuration.total_time_ns', index=1,
      number=2, type=4, cpp_type=4, label=1,
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
  serialized_start=67,
  serialized_end=119,
)


_STATUSREQUESTSTATS = _descriptor.Descriptor(
  name='StatusRequestStats',
  full_name='nvidia.inferenceserver.StatusRequestStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='nvidia.inferenceserver.StatusRequestStats.success', index=0,
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
  serialized_start=121,
  serialized_end=196,
)


_PROFILEREQUESTSTATS = _descriptor.Descriptor(
  name='ProfileRequestStats',
  full_name='nvidia.inferenceserver.ProfileRequestStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='nvidia.inferenceserver.ProfileRequestStats.success', index=0,
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
  serialized_start=198,
  serialized_end=274,
)


_HEALTHREQUESTSTATS = _descriptor.Descriptor(
  name='HealthRequestStats',
  full_name='nvidia.inferenceserver.HealthRequestStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='nvidia.inferenceserver.HealthRequestStats.success', index=0,
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
  serialized_start=276,
  serialized_end=351,
)


_INFERREQUESTSTATS = _descriptor.Descriptor(
  name='InferRequestStats',
  full_name='nvidia.inferenceserver.InferRequestStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='nvidia.inferenceserver.InferRequestStats.success', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='failed', full_name='nvidia.inferenceserver.InferRequestStats.failed', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='compute', full_name='nvidia.inferenceserver.InferRequestStats.compute', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='queue', full_name='nvidia.inferenceserver.InferRequestStats.queue', index=3,
      number=4, type=11, cpp_type=10, label=1,
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
  serialized_start=354,
  serialized_end=590,
)


_MODELVERSIONSTATUS_INFERSTATSENTRY = _descriptor.Descriptor(
  name='InferStatsEntry',
  full_name='nvidia.inferenceserver.ModelVersionStatus.InferStatsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='nvidia.inferenceserver.ModelVersionStatus.InferStatsEntry.key', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='nvidia.inferenceserver.ModelVersionStatus.InferStatsEntry.value', index=1,
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
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=820,
  serialized_end=912,
)

_MODELVERSIONSTATUS = _descriptor.Descriptor(
  name='ModelVersionStatus',
  full_name='nvidia.inferenceserver.ModelVersionStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ready_state', full_name='nvidia.inferenceserver.ModelVersionStatus.ready_state', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='infer_stats', full_name='nvidia.inferenceserver.ModelVersionStatus.infer_stats', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_execution_count', full_name='nvidia.inferenceserver.ModelVersionStatus.model_execution_count', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_inference_count', full_name='nvidia.inferenceserver.ModelVersionStatus.model_inference_count', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_MODELVERSIONSTATUS_INFERSTATSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=593,
  serialized_end=912,
)


_MODELSTATUS_VERSIONSTATUSENTRY = _descriptor.Descriptor(
  name='VersionStatusEntry',
  full_name='nvidia.inferenceserver.ModelStatus.VersionStatusEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='nvidia.inferenceserver.ModelStatus.VersionStatusEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='nvidia.inferenceserver.ModelStatus.VersionStatusEntry.value', index=1,
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
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1063,
  serialized_end=1159,
)

_MODELSTATUS = _descriptor.Descriptor(
  name='ModelStatus',
  full_name='nvidia.inferenceserver.ModelStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='config', full_name='nvidia.inferenceserver.ModelStatus.config', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version_status', full_name='nvidia.inferenceserver.ModelStatus.version_status', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_MODELSTATUS_VERSIONSTATUSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=915,
  serialized_end=1159,
)


_SERVERSTATUS_MODELSTATUSENTRY = _descriptor.Descriptor(
  name='ModelStatusEntry',
  full_name='nvidia.inferenceserver.ServerStatus.ModelStatusEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='nvidia.inferenceserver.ServerStatus.ModelStatusEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='nvidia.inferenceserver.ServerStatus.ModelStatusEntry.value', index=1,
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
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1566,
  serialized_end=1653,
)

_SERVERSTATUS = _descriptor.Descriptor(
  name='ServerStatus',
  full_name='nvidia.inferenceserver.ServerStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='nvidia.inferenceserver.ServerStatus.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='nvidia.inferenceserver.ServerStatus.version', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ready_state', full_name='nvidia.inferenceserver.ServerStatus.ready_state', index=2,
      number=7, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='uptime_ns', full_name='nvidia.inferenceserver.ServerStatus.uptime_ns', index=3,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_status', full_name='nvidia.inferenceserver.ServerStatus.model_status', index=4,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status_stats', full_name='nvidia.inferenceserver.ServerStatus.status_stats', index=5,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='profile_stats', full_name='nvidia.inferenceserver.ServerStatus.profile_stats', index=6,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='health_stats', full_name='nvidia.inferenceserver.ServerStatus.health_stats', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_SERVERSTATUS_MODELSTATUSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1162,
  serialized_end=1653,
)

_STATUSREQUESTSTATS.fields_by_name['success'].message_type = _STATDURATION
_PROFILEREQUESTSTATS.fields_by_name['success'].message_type = _STATDURATION
_HEALTHREQUESTSTATS.fields_by_name['success'].message_type = _STATDURATION
_INFERREQUESTSTATS.fields_by_name['success'].message_type = _STATDURATION
_INFERREQUESTSTATS.fields_by_name['failed'].message_type = _STATDURATION
_INFERREQUESTSTATS.fields_by_name['compute'].message_type = _STATDURATION
_INFERREQUESTSTATS.fields_by_name['queue'].message_type = _STATDURATION
_MODELVERSIONSTATUS_INFERSTATSENTRY.fields_by_name['value'].message_type = _INFERREQUESTSTATS
_MODELVERSIONSTATUS_INFERSTATSENTRY.containing_type = _MODELVERSIONSTATUS
_MODELVERSIONSTATUS.fields_by_name['ready_state'].enum_type = _MODELREADYSTATE
_MODELVERSIONSTATUS.fields_by_name['infer_stats'].message_type = _MODELVERSIONSTATUS_INFERSTATSENTRY
_MODELSTATUS_VERSIONSTATUSENTRY.fields_by_name['value'].message_type = _MODELVERSIONSTATUS
_MODELSTATUS_VERSIONSTATUSENTRY.containing_type = _MODELSTATUS
_MODELSTATUS.fields_by_name['config'].message_type = model__config__pb2._MODELCONFIG
_MODELSTATUS.fields_by_name['version_status'].message_type = _MODELSTATUS_VERSIONSTATUSENTRY
_SERVERSTATUS_MODELSTATUSENTRY.fields_by_name['value'].message_type = _MODELSTATUS
_SERVERSTATUS_MODELSTATUSENTRY.containing_type = _SERVERSTATUS
_SERVERSTATUS.fields_by_name['ready_state'].enum_type = _SERVERREADYSTATE
_SERVERSTATUS.fields_by_name['model_status'].message_type = _SERVERSTATUS_MODELSTATUSENTRY
_SERVERSTATUS.fields_by_name['status_stats'].message_type = _STATUSREQUESTSTATS
_SERVERSTATUS.fields_by_name['profile_stats'].message_type = _PROFILEREQUESTSTATS
_SERVERSTATUS.fields_by_name['health_stats'].message_type = _HEALTHREQUESTSTATS
DESCRIPTOR.message_types_by_name['StatDuration'] = _STATDURATION
DESCRIPTOR.message_types_by_name['StatusRequestStats'] = _STATUSREQUESTSTATS
DESCRIPTOR.message_types_by_name['ProfileRequestStats'] = _PROFILEREQUESTSTATS
DESCRIPTOR.message_types_by_name['HealthRequestStats'] = _HEALTHREQUESTSTATS
DESCRIPTOR.message_types_by_name['InferRequestStats'] = _INFERREQUESTSTATS
DESCRIPTOR.message_types_by_name['ModelVersionStatus'] = _MODELVERSIONSTATUS
DESCRIPTOR.message_types_by_name['ModelStatus'] = _MODELSTATUS
DESCRIPTOR.message_types_by_name['ServerStatus'] = _SERVERSTATUS
DESCRIPTOR.enum_types_by_name['ModelReadyState'] = _MODELREADYSTATE
DESCRIPTOR.enum_types_by_name['ServerReadyState'] = _SERVERREADYSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StatDuration = _reflection.GeneratedProtocolMessageType('StatDuration', (_message.Message,), dict(
  DESCRIPTOR = _STATDURATION,
  __module__ = 'server_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.StatDuration)
  ))
_sym_db.RegisterMessage(StatDuration)

StatusRequestStats = _reflection.GeneratedProtocolMessageType('StatusRequestStats', (_message.Message,), dict(
  DESCRIPTOR = _STATUSREQUESTSTATS,
  __module__ = 'server_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.StatusRequestStats)
  ))
_sym_db.RegisterMessage(StatusRequestStats)

ProfileRequestStats = _reflection.GeneratedProtocolMessageType('ProfileRequestStats', (_message.Message,), dict(
  DESCRIPTOR = _PROFILEREQUESTSTATS,
  __module__ = 'server_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ProfileRequestStats)
  ))
_sym_db.RegisterMessage(ProfileRequestStats)

HealthRequestStats = _reflection.GeneratedProtocolMessageType('HealthRequestStats', (_message.Message,), dict(
  DESCRIPTOR = _HEALTHREQUESTSTATS,
  __module__ = 'server_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.HealthRequestStats)
  ))
_sym_db.RegisterMessage(HealthRequestStats)

InferRequestStats = _reflection.GeneratedProtocolMessageType('InferRequestStats', (_message.Message,), dict(
  DESCRIPTOR = _INFERREQUESTSTATS,
  __module__ = 'server_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferRequestStats)
  ))
_sym_db.RegisterMessage(InferRequestStats)

ModelVersionStatus = _reflection.GeneratedProtocolMessageType('ModelVersionStatus', (_message.Message,), dict(

  InferStatsEntry = _reflection.GeneratedProtocolMessageType('InferStatsEntry', (_message.Message,), dict(
    DESCRIPTOR = _MODELVERSIONSTATUS_INFERSTATSENTRY,
    __module__ = 'server_status_pb2'
    # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ModelVersionStatus.InferStatsEntry)
    ))
  ,
  DESCRIPTOR = _MODELVERSIONSTATUS,
  __module__ = 'server_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ModelVersionStatus)
  ))
_sym_db.RegisterMessage(ModelVersionStatus)
_sym_db.RegisterMessage(ModelVersionStatus.InferStatsEntry)

ModelStatus = _reflection.GeneratedProtocolMessageType('ModelStatus', (_message.Message,), dict(

  VersionStatusEntry = _reflection.GeneratedProtocolMessageType('VersionStatusEntry', (_message.Message,), dict(
    DESCRIPTOR = _MODELSTATUS_VERSIONSTATUSENTRY,
    __module__ = 'server_status_pb2'
    # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ModelStatus.VersionStatusEntry)
    ))
  ,
  DESCRIPTOR = _MODELSTATUS,
  __module__ = 'server_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ModelStatus)
  ))
_sym_db.RegisterMessage(ModelStatus)
_sym_db.RegisterMessage(ModelStatus.VersionStatusEntry)

ServerStatus = _reflection.GeneratedProtocolMessageType('ServerStatus', (_message.Message,), dict(

  ModelStatusEntry = _reflection.GeneratedProtocolMessageType('ModelStatusEntry', (_message.Message,), dict(
    DESCRIPTOR = _SERVERSTATUS_MODELSTATUSENTRY,
    __module__ = 'server_status_pb2'
    # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ServerStatus.ModelStatusEntry)
    ))
  ,
  DESCRIPTOR = _SERVERSTATUS,
  __module__ = 'server_status_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.ServerStatus)
  ))
_sym_db.RegisterMessage(ServerStatus)
_sym_db.RegisterMessage(ServerStatus.ModelStatusEntry)


_MODELVERSIONSTATUS_INFERSTATSENTRY._options = None
_MODELSTATUS_VERSIONSTATUSENTRY._options = None
_SERVERSTATUS_MODELSTATUSENTRY._options = None
# @@protoc_insertion_point(module_scope)
