"""General settings of SQcircuit."""

# accuracy settings related to computation in SQcircuit algorithm
ACC = {
    'sing_mode_detect': 1e-11,  # singular mode detection accuracy
    'Gram–Schmidt': 1e-6,  # Gram–Schmidt process accuracy
    'har_mode_elim': 1e-11,  # harmonic mode elimination accuracy
}

# General flag which states that SQcircuit is in optimization mode.
_OPTIM_MODE = False
_ENG = 'numpy'

def set_engine(eng: str) -> None:
    """Choose the internal numerical package for ``SQcircuit` to use.  This
    sets the type of objects used for element values, computed eigenfrequencies
    and eigenvectors, etc.

    - ``'NumPy'``
    - ``'PyTorch'`` (allows gradient-based optimization)
    
    Parameters
    ----------
    eng:
        Numerical engine for SQcircuit to use.
    """
    global _ENG

    eng = eng.lower()

    if eng == 'numpy':
        set_optim_mode(False)
    elif eng == 'pytorch':
        set_optim_mode(True)
    else:
        raise ValueError("Invalid engine passed. Currently SQcircuit supports 'NumPy' and 'PyTorch'")

    _ENG = eng

def get_engine() -> str:
    """Get the internal numerical engine which ``SQcircuit`` is using.

    Returns
    ----------
        The numerical engine currently in use.
    """
    return _ENG


def set_optim_mode(s: bool) -> None:
    """Utility function for setting engine of ``SQcircuit``. If the optimization
    mode is ``True``, then ``SQcircuit`` uses ``PyTorch``; if it is false
    it uses ``NumPy``.

    Parameters
    ----------
    s:
        State of the optimization mode as boolean variable.
    """

    global _OPTIM_MODE

    _OPTIM_MODE = s


def get_optim_mode() -> bool:
    """Utility function for getting the current engine of ``SQcircuit``. If
    the engine is ``NumPy``, the optimization mode is ``False``; if it is
    ``PyTorch``, the optimization mode is ``True``.
    
    Returns
    ----------
        If the numerical engine is NumPy (``False``) or PyTorch (``True``)
    """

    return _OPTIM_MODE
