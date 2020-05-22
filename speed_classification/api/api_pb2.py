# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: api.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='api.proto',
  package='nvidia.inferenceserver',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\tapi.proto\x12\x16nvidia.inferenceserver\"\xda\x03\n\x12InferRequestHeader\x12\n\n\x02id\x18\x05 \x01(\x04\x12\r\n\x05\x66lags\x18\x06 \x01(\r\x12\x16\n\x0e\x63orrelation_id\x18\x04 \x01(\x04\x12\x12\n\nbatch_size\x18\x01 \x01(\r\x12?\n\x05input\x18\x02 \x03(\x0b\x32\x30.nvidia.inferenceserver.InferRequestHeader.Input\x12\x41\n\x06output\x18\x03 \x03(\x0b\x32\x31.nvidia.inferenceserver.InferRequestHeader.Output\x1a<\n\x05Input\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x64ims\x18\x02 \x03(\x03\x12\x17\n\x0f\x62\x61tch_byte_size\x18\x03 \x01(\x04\x1at\n\x06Output\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x44\n\x03\x63ls\x18\x03 \x01(\x0b\x32\x37.nvidia.inferenceserver.InferRequestHeader.Output.Class\x1a\x16\n\x05\x43lass\x12\r\n\x05\x63ount\x18\x01 \x01(\r\"E\n\x04\x46lag\x12\r\n\tFLAG_NONE\x10\x00\x12\x17\n\x13\x46LAG_SEQUENCE_START\x10\x01\x12\x15\n\x11\x46LAG_SEQUENCE_END\x10\x02\"\x89\x04\n\x13InferResponseHeader\x12\n\n\x02id\x18\x05 \x01(\x04\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x15\n\rmodel_version\x18\x02 \x01(\x03\x12\x12\n\nbatch_size\x18\x03 \x01(\r\x12\x42\n\x06output\x18\x04 \x03(\x0b\x32\x32.nvidia.inferenceserver.InferResponseHeader.Output\x1a\xe2\x02\n\x06Output\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x43\n\x03raw\x18\x02 \x01(\x0b\x32\x36.nvidia.inferenceserver.InferResponseHeader.Output.Raw\x12Q\n\rbatch_classes\x18\x03 \x03(\x0b\x32:.nvidia.inferenceserver.InferResponseHeader.Output.Classes\x1a,\n\x03Raw\x12\x0c\n\x04\x64ims\x18\x01 \x03(\x03\x12\x17\n\x0f\x62\x61tch_byte_size\x18\x02 \x01(\x04\x1a\x32\n\x05\x43lass\x12\x0b\n\x03idx\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x02\x12\r\n\x05label\x18\x03 \x01(\t\x1aP\n\x07\x43lasses\x12\x45\n\x03\x63ls\x18\x01 \x03(\x0b\x32\x38.nvidia.inferenceserver.InferResponseHeader.Output.Classb\x06proto3')
)



_INFERREQUESTHEADER_FLAG = _descriptor.EnumDescriptor(
  name='Flag',
  full_name='nvidia.inferenceserver.InferRequestHeader.Flag',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FLAG_NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLAG_SEQUENCE_START', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLAG_SEQUENCE_END', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=443,
  serialized_end=512,
)
_sym_db.RegisterEnumDescriptor(_INFERREQUESTHEADER_FLAG)


_INFERREQUESTHEADER_INPUT = _descriptor.Descriptor(
  name='Input',
  full_name='nvidia.inferenceserver.InferRequestHeader.Input',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='nvidia.inferenceserver.InferRequestHeader.Input.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dims', full_name='nvidia.inferenceserver.InferRequestHeader.Input.dims', index=1,
      number=2, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_byte_size', full_name='nvidia.inferenceserver.InferRequestHeader.Input.batch_byte_size', index=2,
      number=3, type=4, cpp_type=4, label=1,
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
  serialized_start=263,
  serialized_end=323,
)

_INFERREQUESTHEADER_OUTPUT_CLASS = _descriptor.Descriptor(
  name='Class',
  full_name='nvidia.inferenceserver.InferRequestHeader.Output.Class',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='count', full_name='nvidia.inferenceserver.InferRequestHeader.Output.Class.count', index=0,
      number=1, type=13, cpp_type=3, label=1,
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
  serialized_start=419,
  serialized_end=441,
)

_INFERREQUESTHEADER_OUTPUT = _descriptor.Descriptor(
  name='Output',
  full_name='nvidia.inferenceserver.InferRequestHeader.Output',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='nvidia.inferenceserver.InferRequestHeader.Output.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cls', full_name='nvidia.inferenceserver.InferRequestHeader.Output.cls', index=1,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_INFERREQUESTHEADER_OUTPUT_CLASS, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=325,
  serialized_end=441,
)

_INFERREQUESTHEADER = _descriptor.Descriptor(
  name='InferRequestHeader',
  full_name='nvidia.inferenceserver.InferRequestHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='nvidia.inferenceserver.InferRequestHeader.id', index=0,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flags', full_name='nvidia.inferenceserver.InferRequestHeader.flags', index=1,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='correlation_id', full_name='nvidia.inferenceserver.InferRequestHeader.correlation_id', index=2,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='nvidia.inferenceserver.InferRequestHeader.batch_size', index=3,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input', full_name='nvidia.inferenceserver.InferRequestHeader.input', index=4,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output', full_name='nvidia.inferenceserver.InferRequestHeader.output', index=5,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_INFERREQUESTHEADER_INPUT, _INFERREQUESTHEADER_OUTPUT, ],
  enum_types=[
    _INFERREQUESTHEADER_FLAG,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=38,
  serialized_end=512,
)


_INFERRESPONSEHEADER_OUTPUT_RAW = _descriptor.Descriptor(
  name='Raw',
  full_name='nvidia.inferenceserver.InferResponseHeader.Output.Raw',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dims', full_name='nvidia.inferenceserver.InferResponseHeader.Output.Raw.dims', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_byte_size', full_name='nvidia.inferenceserver.InferResponseHeader.Output.Raw.batch_byte_size', index=1,
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
  serialized_start=858,
  serialized_end=902,
)

_INFERRESPONSEHEADER_OUTPUT_CLASS = _descriptor.Descriptor(
  name='Class',
  full_name='nvidia.inferenceserver.InferResponseHeader.Output.Class',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='idx', full_name='nvidia.inferenceserver.InferResponseHeader.Output.Class.idx', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='nvidia.inferenceserver.InferResponseHeader.Output.Class.value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label', full_name='nvidia.inferenceserver.InferResponseHeader.Output.Class.label', index=2,
      number=3, type=9, cpp_type=9, label=1,
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
  serialized_start=904,
  serialized_end=954,
)

_INFERRESPONSEHEADER_OUTPUT_CLASSES = _descriptor.Descriptor(
  name='Classes',
  full_name='nvidia.inferenceserver.InferResponseHeader.Output.Classes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cls', full_name='nvidia.inferenceserver.InferResponseHeader.Output.Classes.cls', index=0,
      number=1, type=11, cpp_type=10, label=3,
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
  serialized_start=956,
  serialized_end=1036,
)

_INFERRESPONSEHEADER_OUTPUT = _descriptor.Descriptor(
  name='Output',
  full_name='nvidia.inferenceserver.InferResponseHeader.Output',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='nvidia.inferenceserver.InferResponseHeader.Output.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='raw', full_name='nvidia.inferenceserver.InferResponseHeader.Output.raw', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_classes', full_name='nvidia.inferenceserver.InferResponseHeader.Output.batch_classes', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_INFERRESPONSEHEADER_OUTPUT_RAW, _INFERRESPONSEHEADER_OUTPUT_CLASS, _INFERRESPONSEHEADER_OUTPUT_CLASSES, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=682,
  serialized_end=1036,
)

_INFERRESPONSEHEADER = _descriptor.Descriptor(
  name='InferResponseHeader',
  full_name='nvidia.inferenceserver.InferResponseHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='nvidia.inferenceserver.InferResponseHeader.id', index=0,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_name', full_name='nvidia.inferenceserver.InferResponseHeader.model_name', index=1,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_version', full_name='nvidia.inferenceserver.InferResponseHeader.model_version', index=2,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='nvidia.inferenceserver.InferResponseHeader.batch_size', index=3,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output', full_name='nvidia.inferenceserver.InferResponseHeader.output', index=4,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_INFERRESPONSEHEADER_OUTPUT, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=515,
  serialized_end=1036,
)

_INFERREQUESTHEADER_INPUT.containing_type = _INFERREQUESTHEADER
_INFERREQUESTHEADER_OUTPUT_CLASS.containing_type = _INFERREQUESTHEADER_OUTPUT
_INFERREQUESTHEADER_OUTPUT.fields_by_name['cls'].message_type = _INFERREQUESTHEADER_OUTPUT_CLASS
_INFERREQUESTHEADER_OUTPUT.containing_type = _INFERREQUESTHEADER
_INFERREQUESTHEADER.fields_by_name['input'].message_type = _INFERREQUESTHEADER_INPUT
_INFERREQUESTHEADER.fields_by_name['output'].message_type = _INFERREQUESTHEADER_OUTPUT
_INFERREQUESTHEADER_FLAG.containing_type = _INFERREQUESTHEADER
_INFERRESPONSEHEADER_OUTPUT_RAW.containing_type = _INFERRESPONSEHEADER_OUTPUT
_INFERRESPONSEHEADER_OUTPUT_CLASS.containing_type = _INFERRESPONSEHEADER_OUTPUT
_INFERRESPONSEHEADER_OUTPUT_CLASSES.fields_by_name['cls'].message_type = _INFERRESPONSEHEADER_OUTPUT_CLASS
_INFERRESPONSEHEADER_OUTPUT_CLASSES.containing_type = _INFERRESPONSEHEADER_OUTPUT
_INFERRESPONSEHEADER_OUTPUT.fields_by_name['raw'].message_type = _INFERRESPONSEHEADER_OUTPUT_RAW
_INFERRESPONSEHEADER_OUTPUT.fields_by_name['batch_classes'].message_type = _INFERRESPONSEHEADER_OUTPUT_CLASSES
_INFERRESPONSEHEADER_OUTPUT.containing_type = _INFERRESPONSEHEADER
_INFERRESPONSEHEADER.fields_by_name['output'].message_type = _INFERRESPONSEHEADER_OUTPUT
DESCRIPTOR.message_types_by_name['InferRequestHeader'] = _INFERREQUESTHEADER
DESCRIPTOR.message_types_by_name['InferResponseHeader'] = _INFERRESPONSEHEADER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InferRequestHeader = _reflection.GeneratedProtocolMessageType('InferRequestHeader', (_message.Message,), dict(

  Input = _reflection.GeneratedProtocolMessageType('Input', (_message.Message,), dict(
    DESCRIPTOR = _INFERREQUESTHEADER_INPUT,
    __module__ = 'api_pb2'
    # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferRequestHeader.Input)
    ))
  ,

  Output = _reflection.GeneratedProtocolMessageType('Output', (_message.Message,), dict(

    Class = _reflection.GeneratedProtocolMessageType('Class', (_message.Message,), dict(
      DESCRIPTOR = _INFERREQUESTHEADER_OUTPUT_CLASS,
      __module__ = 'api_pb2'
      # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferRequestHeader.Output.Class)
      ))
    ,
    DESCRIPTOR = _INFERREQUESTHEADER_OUTPUT,
    __module__ = 'api_pb2'
    # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferRequestHeader.Output)
    ))
  ,
  DESCRIPTOR = _INFERREQUESTHEADER,
  __module__ = 'api_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferRequestHeader)
  ))
_sym_db.RegisterMessage(InferRequestHeader)
_sym_db.RegisterMessage(InferRequestHeader.Input)
_sym_db.RegisterMessage(InferRequestHeader.Output)
_sym_db.RegisterMessage(InferRequestHeader.Output.Class)

InferResponseHeader = _reflection.GeneratedProtocolMessageType('InferResponseHeader', (_message.Message,), dict(

  Output = _reflection.GeneratedProtocolMessageType('Output', (_message.Message,), dict(

    Raw = _reflection.GeneratedProtocolMessageType('Raw', (_message.Message,), dict(
      DESCRIPTOR = _INFERRESPONSEHEADER_OUTPUT_RAW,
      __module__ = 'api_pb2'
      # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferResponseHeader.Output.Raw)
      ))
    ,

    Class = _reflection.GeneratedProtocolMessageType('Class', (_message.Message,), dict(
      DESCRIPTOR = _INFERRESPONSEHEADER_OUTPUT_CLASS,
      __module__ = 'api_pb2'
      # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferResponseHeader.Output.Class)
      ))
    ,

    Classes = _reflection.GeneratedProtocolMessageType('Classes', (_message.Message,), dict(
      DESCRIPTOR = _INFERRESPONSEHEADER_OUTPUT_CLASSES,
      __module__ = 'api_pb2'
      # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferResponseHeader.Output.Classes)
      ))
    ,
    DESCRIPTOR = _INFERRESPONSEHEADER_OUTPUT,
    __module__ = 'api_pb2'
    # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferResponseHeader.Output)
    ))
  ,
  DESCRIPTOR = _INFERRESPONSEHEADER,
  __module__ = 'api_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.inferenceserver.InferResponseHeader)
  ))
_sym_db.RegisterMessage(InferResponseHeader)
_sym_db.RegisterMessage(InferResponseHeader.Output)
_sym_db.RegisterMessage(InferResponseHeader.Output.Raw)
_sym_db.RegisterMessage(InferResponseHeader.Output.Class)
_sym_db.RegisterMessage(InferResponseHeader.Output.Classes)


# @@protoc_insertion_point(module_scope)
