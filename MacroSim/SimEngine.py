from typing import Union

import numpy as np
from Logger import Logger
from IS_LM import IS_LM


class Engine:

    def __init__(self, sr_model: Union[IS_LM, ...], lr_model: ..., entropy_variance: int = 1, random_seed: int = 0):
        self.logger = Logger()

        self.sr_model = sr_model
        self.lr_model = lr_model

        # scale represents standard deviation, __init__ takes variance to follow normal distribution notation conventions
        self.entropy_dist = np.random.normal(loc=0, scale=np.sqrt(entropy_variance), size=self.sr_model.shape)

    def simulate(self, steps: int) -> None:
        ...