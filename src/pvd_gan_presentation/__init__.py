# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'pvd-gan-presentation'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


from .model import (
    PVDDiscriminator,
    PVDGenerator,
)

from .data import (
    sample_data
)

from .train import (
    PVDGAN
)
