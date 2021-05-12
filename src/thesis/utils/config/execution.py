from .base import Config
from typing import Dict


class RunConfig(Config):
    """Configuration class for Runs management:
    - name `run_name` (string), is the experiment name
    - number `run_number` (number), is the number for the relative experiment
    - iteration `run_iteration` (string), is the iteration code
    - seed `seed` (number), is the seed for the system randomness
    """

    def __init__(self, *args, **kwargs):
        # inizialize all variables
        self.name: str = kwargs["run_name"]  # (string) experiment name
        self.number: int = kwargs["run_number"]  # (number) for the relative experiment
        self.iteration: str = kwargs["run_iteration"]  # (string) iteration code
        self.seed: int = kwargs["seed"]  # (number) the seed for the system randomness

    def get_fingerprint(self) -> Dict:
        # return disctionay of important value to hash
        return dict({"seed": self.seed})


class LogConfig(Config):
    """Configuration class for Logging management:
    - verbose `verbose` (bool), flag for logging information logs
    - debug `debug` (bool), flag for logging all debug logs ( ⚠️ tons of logs ⚠️ )
    - time `time` (bool), flag for logging time values of functions (when supported)
    - callback `callback` (string), is the callback code name (`unused`)
    """

    from logging import Logger

    logger: Logger = None

    def __init__(self, *args, **kwargs):
        # inizialize all variables
        self.verbose: bool = kwargs["verbose"]  # (bool) flag
        self.debug: bool = kwargs["debug"]  # (bool) flag
        self.time: bool = kwargs["time"]  # (bool) flag
        self.callback: str = kwargs["callback"]  # (string) call back setting

    def set_logger(self, logger: Logger):
        self.logger = logger

    def get_fingerprint(self) -> Dict:
        # return disctionay of important value to hash
        return dict(
            {
                # nothing to hash here
            }
        )
