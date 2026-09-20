"""
Small helpers shared across the smoothing, antenna-mapping, and
lag-analysis stages.

None of these functions touch casacore directly, so they can be
imported and unit-tested without a working CASA/casacore install.
"""

import re


def natural_sort_key(text):
    """
    Sort strings naturally, so 'scan2' sorts before 'scan10'.

    Use as the `key=` argument to `sorted()`.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


def safe_filename(text):
    """
    Replace filesystem-unfriendly characters with underscores, so a
    value like an antenna name can be used directly as a path
    component.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def normalise_name(value):
    """
    Convert a casacore string/bytes value into a plain Python string.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def identify_pol_axis(shape, max_pols=4):
    """
    Work out which axis of a 2D CPARAM/FLAG cell is polarization.

    CASA calibration tables store CPARAM cells as either

        (nchan, npol)   or   (npol, nchan)

    Distinguish the two by assuming the polarization axis is never
    longer than `max_pols` (in practice at most 4: RR, RL, LR, LL or
    XX, XY, YX, YY).
    """
    if len(shape) != 2:
        raise RuntimeError(
            "Expected a 2D CPARAM/FLAG cell, got shape {}".format(shape)
        )

    if shape[1] <= max_pols:
        return 1

    if shape[0] <= max_pols:
        return 0

    raise RuntimeError(
        "Could not identify the polarization axis for shape {} "
        "(neither dimension is <= {})".format(shape, max_pols)
    )


def channel_and_pol_count(shape):
    """
    Return (nchan, npol) for a 2D CPARAM/FLAG cell, regardless of
    whether it is stored as (nchan, npol) or (npol, nchan).
    """
    pol_axis = identify_pol_axis(shape)

    if pol_axis == 1:
        return shape[0], shape[1]

    return shape[1], shape[0]


def get_pol_spectrum(cparam, flags, pol):
    """
    Return (spectrum, flag) for one polarization, from either the
    (nchan, npol) or (npol, nchan) CPARAM/FLAG layout.
    """
    pol_axis = identify_pol_axis(cparam.shape)

    npol = cparam.shape[pol_axis]

    if pol >= npol:
        raise RuntimeError(
            "Requested polarization {} but only {} present "
            "(shape {})".format(pol, npol, cparam.shape)
        )

    if pol_axis == 1:
        return cparam[:, pol].copy(), flags[:, pol].copy()

    return cparam[pol, :].copy(), flags[pol, :].copy()


def iter_pol_spectra(cparam, flags):
    """
    Yield (pol_index, spectrum, flag) for every polarization present
    in a 2D CPARAM/FLAG cell, regardless of axis layout.
    """
    pol_axis = identify_pol_axis(cparam.shape)

    npol = cparam.shape[pol_axis]

    for pol in range(npol):
        if pol_axis == 1:
            yield pol, cparam[:, pol].copy(), flags[:, pol].copy()
        else:
            yield pol, cparam[pol, :].copy(), flags[pol, :].copy()


def set_pol_spectrum(cparam, pol, spectrum):
    """
    Write one polarization's spectrum back into a CPARAM cell,
    in place, regardless of axis layout.
    """
    pol_axis = identify_pol_axis(cparam.shape)

    if pol_axis == 1:
        cparam[:, pol] = spectrum
    else:
        cparam[pol, :] = spectrum
