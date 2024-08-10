from grandalf.utils.poset import Poset


def test_poset():
    import pytest
    poset = Poset(())
    poset.add(3)
    poset.add(1)
    poset.add(2)

    assert list(iter(poset)) == [3, 1, 2]

    poset.remove(1)
    assert list(iter(poset)) == [3, 2]
    assert str(poset) == '0.| 3\n1.| 2'.replace('\r\n', '\n').replace('\r', '\n')

    poset.add(4)
    assert list(iter(poset)) == [3, 2, 4]

    assert poset.index(2) == 1

    assert poset.get(2) == 2
    assert poset.get(5) is None

    assert poset[0] == 3
    assert poset[1] == 2

    with pytest.raises(IndexError):
        poset[22]

    assert len(poset) == 3

    cp = poset.copy()
    assert cp == poset
    assert list(iter(cp)) == [3, 2, 4]

    cp.remove(3)
    assert cp != poset
    cp.add(3)

    # A bit weird: equality doesn't depend on order (eq considers it unsorted).
    assert cp == poset
    assert not cp != poset

    cp.add(17)
    cp.add(9)
    assert list(iter(cp)) == [2, 4, 3, 17, 9]
    assert list(iter(poset)) == [3, 2, 4]

    poset.update(cp)
    assert list(iter(poset)) == [3, 2, 4, 17, 9]
