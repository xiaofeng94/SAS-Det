# Copyright (c) Facebook, Inc. and its affiliates.
from . import ovd_register as _ovd_register  # ensure the builtin datasets are registered

__all__ = [k for k in globals().keys() if not k.startswith("_")]
