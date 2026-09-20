import numpy as np
import pytest

from bandpass_stability.caltable_utils import (
    channel_and_pol_count,
    get_pol_spectrum,
    identify_pol_axis,
    iter_pol_spectra,
    natural_sort_key,
    safe_filename,
    set_pol_spectrum,
)


def test_natural_sort_key_orders_numbers_numerically():
    names = ["scan10", "scan2", "scan1"]
    assert sorted(names, key=natural_sort_key) == ["scan1", "scan2", "scan10"]


def test_safe_filename_replaces_awkward_characters():
    assert safe_filename("Ant/01 (east)") == "Ant_01_east_"


def test_identify_pol_axis_nchan_npol_layout():
    # (nchan, npol) = (64, 2)
    assert identify_pol_axis((64, 2)) == 1


def test_identify_pol_axis_npol_nchan_layout():
    # (npol, nchan) = (2, 64)
    assert identify_pol_axis((2, 64)) == 0


def test_identify_pol_axis_raises_when_ambiguous():
    with pytest.raises(RuntimeError):
        identify_pol_axis((8, 8), max_pols=4)


def test_identify_pol_axis_requires_2d():
    with pytest.raises(RuntimeError):
        identify_pol_axis((64,))


def test_channel_and_pol_count_both_layouts():
    assert channel_and_pol_count((64, 2)) == (64, 2)
    assert channel_and_pol_count((2, 64)) == (64, 2)


def test_get_pol_spectrum_matches_across_layouts():
    nchan, npol = 8, 2
    rng = np.random.default_rng(0)
    values = rng.normal(size=(nchan, npol)) + 1j * rng.normal(size=(nchan, npol))
    flags = np.zeros((nchan, npol), dtype=bool)

    spectrum_a, flag_a = get_pol_spectrum(values, flags, pol=1)

    # Same data, transposed to the (npol, nchan) layout, should give
    # the same answer for the same physical polarization.
    spectrum_b, flag_b = get_pol_spectrum(values.T, flags.T, pol=1)

    np.testing.assert_allclose(spectrum_a, spectrum_b)
    np.testing.assert_array_equal(flag_a, flag_b)


def test_get_pol_spectrum_rejects_out_of_range_pol():
    values = np.zeros((8, 2), dtype=complex)
    flags = np.zeros((8, 2), dtype=bool)

    with pytest.raises(RuntimeError):
        get_pol_spectrum(values, flags, pol=5)


def test_iter_pol_spectra_yields_every_polarization():
    nchan, npol = 8, 2
    values = np.arange(nchan * npol).reshape(nchan, npol).astype(complex)
    flags = np.zeros((nchan, npol), dtype=bool)

    seen = list(iter_pol_spectra(values, flags))

    assert [pol for pol, _, _ in seen] == [0, 1]
    np.testing.assert_allclose(seen[0][1], values[:, 0])
    np.testing.assert_allclose(seen[1][1], values[:, 1])


def test_set_pol_spectrum_round_trips_through_get_pol_spectrum():
    nchan, npol = 8, 2
    cparam = np.zeros((nchan, npol), dtype=complex)
    flags = np.zeros((nchan, npol), dtype=bool)

    new_spectrum = np.arange(nchan).astype(complex)
    set_pol_spectrum(cparam, pol=1, spectrum=new_spectrum)

    round_tripped, _ = get_pol_spectrum(cparam, flags, pol=1)
    np.testing.assert_allclose(round_tripped, new_spectrum)
