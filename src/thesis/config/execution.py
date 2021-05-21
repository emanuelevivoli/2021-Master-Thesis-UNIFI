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
        # (number) for the relative experiment
        self.number: int = int(kwargs["run_number"])
        # (string) iteration code
        self.iteration: str = kwargs["run_iteration"]
        # (number) the seed for the system randomness
        self.seed: int = int(kwargs["seed"])

    def get_fingerprint(self) -> Dict:
        # return disctionay of important value to hash
        return dict({"seed": self.seed})


class LogConfig(Config):
    """Configuration class for Logging management:
    - verbose `verbose` (bool), flag for logging information logs
    - debug `debug` (bool), flag for logging all debug logs ( ⚠️ tons of logs ⚠️ )
    - time `time` (bool), flag for logging time values of functions (when supported)
    - callbacks `callback` (string), is the callback code name (`unused`)
    """

    from logging import Logger

    logger: Logger = None

    def __init__(self, *args, **kwargs):
        # inizialize all variables
        self.verbose: bool = kwargs["verbose"]  # (bool) flag
        self.debug: bool = kwargs["debug_log"]  # (bool) flag
        self.time: bool = kwargs["time"]  # (bool) flag
        # (string) call back setting
        self.callbacks: List[str] = self.str_to_list(kwargs["callbacks"])

    def set_logger(self, logger: Logger):
        self.logger = logger

    def get_fingerprint(self) -> Dict:
        # return disctionay of important value to hash
        return dict(
            {
                # nothing to hash here
            }
        )
