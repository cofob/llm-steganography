import os
import stat
from pathlib import Path

import pytest

from llm_steganography.cli import main
from llm_steganography.constants import KEY_SIZE
from llm_steganography.crypto import generate_key, validate_key
from llm_steganography.errors import InvalidKeyError


def test_generate_key() -> None:
    first = generate_key()
    second = generate_key()
    assert len(first) == KEY_SIZE
    assert first != second


@pytest.mark.parametrize("size", [0, 1, 31, 33])
def test_reject_invalid_key_size(size: int) -> None:
    with pytest.raises(InvalidKeyError):
        validate_key(bytes(size))


def test_keygen_cli_uses_private_permissions(tmp_path: Path) -> None:
    path = tmp_path / "key"
    assert main(["keygen", "-o", str(path)]) == 0
    assert len(path.read_bytes()) == KEY_SIZE
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
