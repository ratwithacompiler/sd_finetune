import sys
import time

import torch

if sys.version_info[:2] >= (3, 9):
    from typing import Union, Optional, Any, TypeVar
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar


class Timed:
    def __init__(self, label: str, seperator_lines: bool = False):
        self.label = label
        self.seperator_lines = seperator_lines
        self.start_ts = self.end_ts = self.secs = None

    def reset(self):
        self.start_ts = None
        self.end_ts = None
        self.secs = None

    def start(self):
        if self.seperator_lines:
            print("=" * 100)
        print(f"TIMED START: {self.label!r}")
        self.start_ts = time.monotonic()
        return self

    def end(self):
        self.end_ts = time.monotonic()
        self.secs = self.end_ts - self.start_ts
        print(f"TIMED DONE : {self.label!r} took {self.secs:.3f} seconds")
        if self.seperator_lines:
            print("=" * 100)

        return self.secs


    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class Timer():
    def __init__(self,
                 print_format: Optional[Union[str, bool]] = None,
                 start: bool = False,
                 run_fn: Optional[Callable[[float], Any]] = None,
                 run_fn_full: Optional[Callable[[float, float, float], Any]] = None,
                 print_fn: Callable[[str], Any] = print,
                 time_fn: Callable[[], float] = time.monotonic,
                 ):
        if isinstance(print_format, bool):
            if print_format is True:
                print_format = "{secs}s"
            else:
                print_format = None

        self.print_format = print_format
        self.run_fn = run_fn
        self.run_fn_full = run_fn_full
        self.print_fn = print_fn
        self.time_fn = time_fn

        # start/end timestamps, by default monotonic time unless time_fn is set
        self.start_ts = None
        self.end_ts = None
        self.seconds: Optional[float] = None

        if start:
            self.start()

    def start(self):
        self.start_ts = self.time_fn()

    def end(self, run: bool = True) -> float:
        self.end_ts = self.time_fn()
        self.seconds = self.end_ts - self.start_ts

        if run:
            self._end()

        return self.seconds

    def _end(self):
        if self.print_format is not None:
            self.print_fn(self.print_format.format(
                self.seconds, self.seconds, self.seconds, self.seconds, self.seconds, self.seconds,
                s = self.seconds, secs = self.seconds, seconds = self.seconds,
                start = self.start_ts, end = self.end_ts,
            ))

        if self.run_fn is not None:
            self.run_fn(self.seconds)

        if self.run_fn_full is not None:
            self.run_fn_full(self.start_ts, self.end_ts, self.seconds)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.end()


def timed_wrapper(fn = None, *, label = None, timed = False):
    def outer(fn):
        def _fn(*args, **kwargs):
            use_label = label or f"Call of {fn.__name__}" + " took {} secs"
            if timed:
                with Timed(use_label):
                    return fn(*args, **kwargs)
            else:
                with Timer(use_label):
                    return fn(*args, **kwargs)

        return _fn

    if fn is None:
        return outer

    return outer(fn)


def _cmp(t1: torch.Tensor, t2: torch.Tensor):
    if t1.shape != t2.shape:
        return False
    if t1.stride != t2.stride:
        return False
    if t1.dtype != t2.dtype:
        return False

    return torch.all(t1.eq(t2)).item()


def _make_cached_caller_targ(target_fn):
    _empty = object()
    cache = [_empty]
    stats = { "hit": 0, "miss": 0, "last": None }

    def _cached_fn(arg):
        if cache[0] is not _empty and _cmp(arg, cache[0][0]):
            stats["hit"] += 1
            stats["last"] = "hit"
            return cache[0][1]

        stats["miss"] += 1
        stats["last"] = "miss"
        res = target_fn(arg)
        cache[0] = (arg, res)
        return res

    return _cached_fn, stats
