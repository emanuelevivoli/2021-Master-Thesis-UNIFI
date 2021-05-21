import hashlib
import os
import pickle
from inspect import signature
from typing import Dict, List
import functools

from thesis.utils.constants import CACHE_DIR


class ConfigWrapper:
    # function_name: str = ''
    # config_dict: Dict = {}
    # config_list: List = []

    def __init__(self, *args, **kwargs):
        """
        eg: 
            ConfigWrapper(
                    `function_name`='function_name', 
                    `a`=el_a, 
                    `b`=el_b, 
                    `c`=el_c, 
                    `d`=el_d, 
                    `el_1`, `el_2`, `el_3`
                )
        """
        self.function_name: str = None
        self.config_dict: Dict = {}
        self.config_list: List = []

        # setting the function name
        # if passed as function_name='function_name' we get it from kwargs
        self.function_name = (
            kwargs.get("function_name") if kwargs.get(
                "function_name", False) else None
        )

        assert (
            self.function_name is not None
        ), f"function_name must be passed as kwargs [eg. function_name='function_name' ]"

        for k, v in kwargs.items():
            self.config_dict[k] = v
            # setattr(self.config_dict, k, v)

        for el in args:
            self.config_list.append(el)

    def __str__(self):
        return (
            "ConfigWrapper(fn="
            + str(self.function_name)
            + " ,cd="
            + str(self.config_dict)
            + " ,cl="
            + str(self.config_list)
            + ")"
        )


def _caching(*conf_args, **conf_kwargs):

    config_object: ConfigWrapper = ConfigWrapper(*conf_args, **conf_kwargs)

    def decorator_caching(function):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            print("creating cache dir")

        @functools.wraps(function)
        def wrapped_function(*args, **kwargs):
            def _obj_to_hex(o):
                if callable(o):
                    return "&".join(
                        [
                            o.__module__,
                            o.__name__,
                            o.__qualname__,
                            str(signature(o)),
                            str(o.__defaults__),
                            str(o.__kwdefaults__),
                        ]
                    )
                elif type(o) == dict:
                    return str(o)
                elif hasattr(o, "__iter__"):
                    l = [str(type(o)), str(len(o)), str(dir(o))]
                    if hasattr(o, "__len__") and len(o) > 0:
                        l.extend([str(list(o)[0]), str(list(o)[-1])])
                    return "&".join(l)
                else:
                    return str(o)

            args_to_hash = (function, config_object)
            hex_digest = hashlib.sha256(
                bytes(str([_obj_to_hex(arg) for arg in args_to_hash]), "utf-8")
            ).hexdigest()

            file_cache = os.path.join(CACHE_DIR, hex_digest)
            if not os.path.exists(file_cache):
                print(
                    f"calculating result for function {function.__qualname__}...",
                    end="",
                )
                result = function(*args, **kwargs)
                with open(file_cache, "wb") as file:
                    pickle.dump(result, file)
                print(
                    f"\rcalculated result for function {function.__qualname__}    ")
            else:
                print(
                    f"loading result from cache for function {function.__qualname__}...",
                    end="",
                )
                with open(file_cache, "rb") as file:
                    result = pickle.load(file)
                print(
                    f"\rloaded result from cache for function {function.__qualname__}    "
                )
            return result

        return wrapped_function

    return decorator_caching


def no_caching(*unused_args, **unused_kwargs):
    def no_decorator_caching(function):
        @functools.wraps(function)
        def no_wrapped_function(*args, **kwargs):
            return function(*args, **kwargs)

        return no_wrapped_function

    return no_decorator_caching


def caching(function, config_object: ConfigWrapper):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print("creating cache dir")

    def wrapped_function(*args, **kwargs):
        def _obj_to_hex(o):
            if callable(o):
                return "&".join(
                    [
                        o.__module__,
                        o.__name__,
                        o.__qualname__,
                        str(signature(o)),
                        str(o.__defaults__),
                        str(o.__kwdefaults__),
                    ]
                )
            elif type(o) == dict:
                return str(o)
            elif hasattr(o, "__iter__"):
                l = [str(type(o)), str(len(o)), str(dir(o))]
                if hasattr(o, "__len__") and len(o) > 0:
                    l.extend([str(list(o)[0]), str(list(o)[-1])])
                return "&".join(l)
            else:
                return str(o)

        args_to_hash = (function, config_object)
        hex_digest = hashlib.sha256(
            bytes(str([_obj_to_hex(arg) for arg in args_to_hash]), "utf-8")
        ).hexdigest()

        file_cache = os.path.join(CACHE_DIR, hex_digest)
        if not os.path.exists(file_cache):
            print(
                f"calculating result for function {function.__qualname__}...", end="")
            result = function(*args, **kwargs)
            with open(file_cache, "wb") as file:
                pickle.dump(result, file)
            print(
                f"\rcalculated result for function {function.__qualname__}    ")
        else:
            print(
                f"loading result from cache for function {function.__qualname__}...",
                end="",
            )
            with open(file_cache, "rb") as file:
                result = pickle.load(file)
            print(
                f"\rloaded result from cache for function {function.__qualname__}    "
            )
        return result

    return wrapped_function
