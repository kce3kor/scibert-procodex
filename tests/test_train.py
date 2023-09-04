import pytest
import os
from pathlib import Path

from scibert.main import *
from scibert.config import CKPTH_DIR, MODEL, EPOCHS


@pytest.mark.train
def test_train():
    # main()

    assert os.path.exists(Path(CKPTH_DIR, f"{MODEL}_{EPOCHS-1}.pt"))
