"""General settings of SQcircuit."""

# accuracy settings related to computation in SQcircuit algorithm
ACC = {
    "sing_mode_detect": 1e-11,  # singular mode detection accuracy
    "Gram–Schmidt": 1e-7,  # Gram–Schmidt process accuracy
    "har_mode_elim": 1e-11,  # harmonic mode elimination accuracy
}

# General flag which states that SQcircuit is in optimization mode.
OPTIM_MODE = False


def set_optim_mode(s: bool) -> None:
    """Set the optimization mode for SQcircuit.

    Parameters
    ----------
    s:
        State of the optim mode as boolean variable.
    """

    global OPTIM_MODE

    OPTIM_MODE = s
