from segmentation.splitting import split_filenames


def test_split_is_reproducible_and_disjoint():
    names = [f"image_{index}.png" for index in range(20)]
    train_a, val_a = split_filenames(names, val_fraction=0.2, seed=123)
    train_b, val_b = split_filenames(names, val_fraction=0.2, seed=123)

    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 16
    assert len(val_a) == 4
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a) | set(val_a) == set(names)
