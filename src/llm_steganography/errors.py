"""Public exception hierarchy."""


class SteganographyError(Exception):
    """Base error for the package."""


class InvalidKeyError(SteganographyError):
    """The key does not have the required format."""


class BlockTooLargeError(SteganographyError):
    """One internal payload block is larger than its frame permits."""


class DecodeError(SteganographyError):
    """The carrier cannot be decoded."""


class TokenizerMismatchError(DecodeError):
    """The carrier does not round-trip through the pinned tokenizer."""


class GenerationError(SteganographyError):
    """The language model cannot produce a valid carrier."""
