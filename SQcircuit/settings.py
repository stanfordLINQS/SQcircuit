"""General settings of SQcircuit."""

# accuracy settings related to computation in SQcircuit algorithm
ACC = {
    "sing_mode_detect": 1e-11,  # singular mode detection accuracy
    "Gram–Schmidt": 1e-7,  # Gram–Schmidt process accuracy
    "har_mode_elim": 1e-11,  # harmonic mode elimination accuracy
}

# General flag which states that SQcircuit is in optimization mode.
_OPTIM_MODE = False

def set_optim_mode(s: bool) -> None:
    """Set the optimization mode for SQcircuit.

    Parameters
    ----------
    s:
        State of the optim mode as boolean variable.
    """

    global _OPTIM_MODE

    _OPTIM_MODE = s


def get_optim_mode() -> bool:

    return _OPTIM_MODE
