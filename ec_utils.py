import re
from functools import lru_cache

def norm_ec(x:str) -> str: # clean EC string 
    return re.sub(r'^\s*EC\s*', '', str(x).strip())

@lru_cache(None)
def ec_levels(ec:str):
    # ECs are of the form X.X.X.X where each X is an integer or '-'
    # Return a tuple of 4 levels, with None for missing levels 
    parts = norm_ec(ec).split('.')
    parts += ['-'] * (4 - len(parts))
    out = []
    for p in parts[:4]:
        if p in ('', '-', None):
            out.append(None)
        else:
            out.append(int(p) if p.isdigit() else None)
    return tuple(out)

@lru_cache(None)
def ec_distance(a:str, b:str) -> int:
    # Compute distance between two EC numbers in the EC hierarchy
    A, B = ec_levels(a), ec_levels(b)
    for i in range(4):
        if A[i] != B[i]:
            # distance = up from A to the lowest common ancestor, then down to B
            return (4 - i) * 2
    return 0

def ec_is_leaf(ec:str) -> bool:
    return all(isinstance(x, int) for x in ec_levels(ec))

# Need this because Problog atoms cannot have '.' or '-' or start with a digit
def r_atom(r: str) -> str: return 'r' + ''.join(ch for ch in str(r) if ch.isdigit())
def g_atom(g:str) -> str: return 'g' + re.sub(r'\W+', '', str(g))
def ec_atom(ec:str) -> str:
    lev = [str(x) for x in ec_levels(ec) if isinstance(x,int)]
    return 'ec_' + '_'.join(lev) if lev else 'ec'
def c_atom(c: str) -> str: return 'c' + re.sub(r'\W+', '', str(c))


def _ec_prefix_tuple(ec: str, depth: int):
    """Tuple key of the first `depth` integer EC levels, consistent with ec_atom/ec_levels."""
    levels = [x for x in ec_levels(ec) if isinstance(x, int)]
    return tuple(levels[:depth])