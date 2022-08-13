from pathlib import Path

from . import hub
from .nn.upstream import S3PRLUpstream, Featurizer

with (Path(__file__).parent / "version.txt").open() as file:
    __version__ = file.read()
