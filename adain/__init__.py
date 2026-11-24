"""AdaIN arbitrary neural style transfer.

* ``adain.model``  - PyTorch networks (training / export). Needs ``torch``.
* ``adain.engine`` - ONNX Runtime inference used by the app. Does not need ``torch``.
* ``adain.ops``    - NumPy implementation of AdaIN and image helpers.
"""

__version__ = "1.0.0"
