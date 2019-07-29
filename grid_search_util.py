import itertools
from typing import Dict, List


def profile_generator(scope: Dict[str, List]) -> List[dict]:
    """
    This method generates individual profiles from the source.
    """
    src = scope.copy()
    for k, v in src.items():
        if type(v) not in [list, tuple]:
            src[k] = [v]
    # gen = list()
    for vals in itertools.product(*list(src.values())):
        # gen.append(dict(zip(src.keys(), vals)))
        yield dict(zip(src.keys(), vals))
