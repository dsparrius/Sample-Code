import pytest
import numpy as np
from project import get_invariant_factors, get_unique_invariant, get_jordan_block, get_jordan_form

#asserts the correct invariant factors are returned
def test_get_invariant_factors():
    assert get_invariant_factors((2,3),(1,2)) == [[(1,1)]]
    assert get_invariant_factors((4,),(2,)) == [[(2,)],[(1,),(1,)]]
    assert get_invariant_factors((3,),(2,)) == [[(1,)]]
    assert get_invariant_factors((3,),(3,)) == []

#asserts ambigus case is handled correctly
def test_get_unique_invariant():
    a = np.array([[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0]])
    b = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
    assert get_unique_invariant(a, [[(2,)],[(1,),(1,)]]) == [[(2,)]]
    assert get_unique_invariant(b, [[(2,)],[(1,),(1,)]]) == [[(1,),(1,)]]

#asserts the correct jordan blocks are returned
def test_get_jordan_block():
    assert (get_jordan_block([1,1]) == np.array([1])).all()
    assert (get_jordan_block([1,2]) == np.array([[1,1],[0,1]])).all()

#asserts correct jordan form is returned
def test_get_jordan_form():
    a = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
    jordan_block1 = np.array([[0,1],[0,0]])
    assert (get_jordan_form((4,),[jordan_block1,jordan_block1]) == a).all()

    b = np.array([[1,0,0],[0,2,0],[0,0,3]])
    assert (get_jordan_form((1,1,1),[np.array([1]),np.array([2]),np.array([3])]) == b).all()