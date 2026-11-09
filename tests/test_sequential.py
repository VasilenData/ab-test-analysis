import numpy as np

from src.sequential import (
    always_valid_p,
    gs_boundaries,
    msprt_normal,
    spend_obf,
    spend_pocock,
)


def test_spending_functions_spend_all_alpha_at_the_end():
    assert abs(float(spend_obf(np.array([1.0]))[0]) - 0.05) < 1e-9
    assert abs(float(spend_pocock(np.array([1.0]))[0]) - 0.05) < 1e-9


def test_obf_spends_little_early_pocock_more():
    assert spend_obf(np.array([0.2]))[0] < 1e-4
    assert spend_pocock(np.array([0.2]))[0] > 0.01


def test_pocock_boundaries_match_textbook():
    b = gs_boundaries(5, spending=spend_pocock)
    assert np.all(np.abs(b - 2.413) < 0.05)           # classic Pocock constant for 5 looks


def test_obf_boundaries_start_high_and_end_near_fixed_sample():
    b = gs_boundaries(5, spending=spend_obf)
    assert b[0] > 4 and np.all(np.diff(b) < 0)
    assert 2.0 < b[-1] < 2.1


def test_boundaries_hold_overall_type_one_error_by_monte_carlo():
    rng = np.random.default_rng(1)
    looks = 5
    b = gs_boundaries(looks, spending=spend_obf)
    paths = np.cumsum(rng.standard_normal((200_000, looks)) / np.sqrt(looks), axis=1)
    z = np.abs(paths) / np.sqrt(np.arange(1, looks + 1) / looks)
    assert abs(float((z > b).any(axis=1).mean()) - 0.05) < 0.004


def test_naive_peeking_inflates_error():
    rng = np.random.default_rng(2)
    looks = 10
    paths = np.cumsum(rng.standard_normal((100_000, looks)) / np.sqrt(looks), axis=1)
    z = np.abs(paths) / np.sqrt(np.arange(1, looks + 1) / looks)
    assert float((z > 1.96).any(axis=1).mean()) > 0.15


def test_msprt_ratio_grows_with_effect_and_is_one_at_zero_information():
    small = msprt_normal(0.001, 1e-4, 0.01)
    big = msprt_normal(0.03, 1e-4, 0.01)
    assert big > small > 0
    assert msprt_normal(0.0, 1e-9, 0.01) < 0.01                 # huge information, zero effect: strong evidence for H0
    assert msprt_normal(0.0, 1e12, 0.01) > 0.99


def test_always_valid_p_never_increases_and_stays_in_unit_interval():
    p = always_valid_p(np.array([0.5, 2.0, 1.0, 40.0, 3.0]))
    assert np.all(np.diff(p) <= 0) and p.max() <= 1 and abs(p[-1] - 0.025) < 1e-12


def test_explicit_equal_fractions_match_integer_looks():
    assert np.allclose(gs_boundaries([0.25, 0.5, 0.75, 1.0]), gs_boundaries(4), atol=1e-9)


def test_unequal_spacing_changes_boundaries_and_keeps_order_of_spending():
    b = gs_boundaries([0.1, 0.3, 1.0], spending=spend_obf)
    assert b[0] > b[1] > b[2] > 1.96
    assert not np.allclose(b, gs_boundaries(3), atol=0.05)


def test_unequal_spacing_holds_type_one_error_by_monte_carlo():
    rng = np.random.default_rng(3)
    t = np.array([0.1, 0.25, 0.6, 1.0])
    for spending in (spend_obf, spend_pocock):
        b = gs_boundaries(t, spending=spending)
        steps = rng.standard_normal((200_000, len(t))) * np.sqrt(np.diff(np.concatenate([[0.0], t])))
        z = np.abs(np.cumsum(steps, axis=1)) / np.sqrt(t)
        assert abs(float((z > b).any(axis=1).mean()) - 0.05) < 0.004


def test_look_before_the_end_spends_less_than_full_alpha():
    b = gs_boundaries([0.5, 0.8], spending=spend_pocock)
    assert b[-1] > 1.96                                   # less than alpha spent in total, so the cutoff is stricter than fixed-sample
    assert np.all(np.isfinite(b))


def test_invalid_look_times_are_refused():
    import pytest
    for bad in ([], [0.0, 1.0], [0.5, 0.5, 1.0], [0.6, 0.4], [0.5, 1.2], [0.5, float("nan")], [[0.5, 1.0]]):
        with pytest.raises(ValueError):
            gs_boundaries(bad)
