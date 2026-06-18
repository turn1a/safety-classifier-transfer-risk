"""Unit tests for the attack-sweep fan-out helpers (pure index logic, no textattack)."""

from transfer_risk.lib.sweep import attack_units, cell_key, shard_key, shard_spans


def test_shard_spans_tile_range_without_gaps_or_overlap() -> None:
    for n, size in [(60, 4), (60, 7), (12, 4), (10, 1), (5, 10), (1, 1)]:
        spans = shard_spans(n, size)
        covered = [i for start, stop in spans for i in range(start, stop)]
        assert covered == list(range(n)), (n, size, spans)
        assert all(stop - start <= max(1, size) for start, stop in spans)
        assert all(start < stop for start, stop in spans)


def test_shard_spans_empty() -> None:
    assert shard_spans(0, 4) == []


def test_shard_spans_coerces_nonpositive_size() -> None:
    assert shard_spans(3, 0) == [(0, 1), (1, 2), (2, 3)]


def test_cell_and_shard_keys() -> None:
    assert cell_key("deberta-base", "bert-attack") == "deberta-base__bert-attack"
    assert shard_key("deberta-base", "bert-attack", 8) == "deberta-base__bert-attack__8"


def test_attack_units_cover_every_surrogate_recipe_shard() -> None:
    surrogates = ["a", "b"]
    recipes = ["pwws", "bae"]
    units = attack_units(surrogates, recipes, n_examples=10, shard_size=4)
    # 2 surrogates * 2 recipes * 3 shards (10 / 4 -> [0,4),[4,8),[8,10))
    assert len(units) == 2 * 2 * 3
    # ordered surrogate, then recipe, then shard start
    assert units[0] == ("a", "pwws", 0, 4)
    assert units[2] == ("a", "pwws", 8, 10)
    # every (surrogate, recipe) cell is fully covered
    for surrogate in surrogates:
        for recipe in recipes:
            cell = [u for u in units if u[0] == surrogate and u[1] == recipe]
            covered = [i for _, _, start, stop in cell for i in range(start, stop)]
            assert covered == list(range(10))
