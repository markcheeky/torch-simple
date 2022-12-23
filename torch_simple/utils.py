from __future__ import annotations

import copy
import io
import pickle
from typing import Any


def deep_copy_with_pickle_fallback(thing: Any) -> Any:
    """
    Creates a copy of the module.
    This was created primarily to copy torch modules, because
    currently not all modules can be copied using deepcopy.
    Problematic are lazy modules, such as LazyBatchNorm1d,
    which can be pickled and unpickled though.
    """
    e1: Exception | None = None
    e2: Exception | None = None

    try:
        return copy.deepcopy(thing)
    except Exception as e:
        e1 = e

    try:
        immediate = io.BytesIO()
        pickle.dump(thing, immediate)
        immediate.seek(0)
        return pickle.load(immediate)
    except Exception as e:
        e2 = e

    raise Exception(
        [
            ValueError("Could not copy object using deepcopy or pickle."),
            e1,
            e2,
        ]
    )
