import pytest

from fau_adapter import prompts as P


def test_builtin_class_lists():
    for spec, n in (("imagenet100", 100), ("imagenet1k", 1000)):
        ps = P.PromptSet.from_classes(spec)
        assert len(ps) == n and ps.templated
        assert all(label.startswith("n") and len(label) == 9 for label in ps.labels)   # WordNet ids
        assert len(set(ps.texts)) == n


def test_imagenet100_is_a_subset_of_imagenet1k():
    small, full = P.PromptSet.from_classes("imagenet100"), P.PromptSet.from_classes("imagenet1k")
    names = dict(zip(full.labels, full.texts))
    assert all(names[label] == text for label, text in zip(small.labels, small.texts))


def test_templates_cycle_and_prompt_ids():
    ps = P.PromptSet(("a", "b"), ("red panda", "hare"), True, "test")
    assert ps.per_group == 8
    assert ps.prompt(1, 0) == "a photo of a hare."
    assert ps.prompt(1, 9) == "a close-up photo of a hare."
    distinct = ps.distinct()
    assert len(distinct) == 16
    for k in range(2):
        for j in range(20):
            assert distinct[ps.prompt_id(k, j)] == ps.prompt(k, j)


def test_free_prompts():
    ps = P.PromptSet(("000000", "000001"), ("a cat on a sofa", "two dogs"), False, "test")
    assert ps.per_group == 1 and ps.prompt(1, 5) == "two dogs" and ps.prompt_id(1, 5) == 1
    assert ps.distinct() == ["a cat on a sofa", "two dogs"]


def test_seeds_of_pairs_never_meet_the_datasets():
    assert P.seed(3, 7) == 3_000_016
    assert P.seed(0, 0, P.PAIR_SEED_OFFSET) == 600_000
    datasets = {P.seed(k, j) for k in range(50) for j in range(1281)}
    pairs = {P.seed(k, j, P.PAIR_SEED_OFFSET) for k in range(50) for j in range(160)}
    assert not datasets & pairs


def test_class_files(tmp_path):
    f = tmp_path / "classes.txt"
    f.write_text("# my classes\nn01\tred panda\n\nhare\n")
    ps = P.PromptSet.from_classes(str(f))
    assert ps.labels == ("n01", "hare") and ps.texts == ("red panda", "hare")
    f.write_text("hare\nhare\n")
    with pytest.raises(ValueError, match="repeated"):
        P.PromptSet.from_classes(str(f))


def test_prompt_files(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("a cat on a sofa\n\n two dogs \n")
    ps = P.PromptSet.from_file(str(f))
    assert ps.texts == ("a cat on a sofa", "two dogs") and not ps.templated
