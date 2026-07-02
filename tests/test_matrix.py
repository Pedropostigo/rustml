import pytest

from rustml import Matrix


def test_construction_and_shape():
    m = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert m.shape == (2, 3)
    assert m[0, 0] == 1.0
    assert m[1, 2] == 6.0


def test_construction_rejects_wrong_length():
    with pytest.raises(ValueError):
        Matrix(2, 2, [1.0, 2.0, 3.0])


def test_zeros():
    m = Matrix.zeros(2, 2)
    assert m.shape == (2, 2)
    assert m[0, 0] == 0.0
    assert m[1, 1] == 0.0


def test_getitem_out_of_bounds():
    m = Matrix.zeros(2, 2)
    with pytest.raises(IndexError):
        m[2, 0]


def test_setitem():
    m = Matrix.zeros(2, 2)
    m[0, 1] = 5.0
    assert m[0, 1] == 5.0


def test_transpose():
    m = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t = m.transpose()
    assert t.shape == (3, 2)
    assert t[0, 0] == 1.0
    assert t[2, 1] == 6.0


def test_add():
    a = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    b = Matrix(2, 2, [5.0, 6.0, 7.0, 8.0])
    c = a + b
    assert c[0, 0] == 6.0
    assert c[1, 1] == 12.0


def test_add_shape_mismatch():
    a = Matrix.zeros(2, 2)
    b = Matrix.zeros(3, 3)
    with pytest.raises(ValueError):
        a + b


def test_scalar_mul():
    a = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    b = a * 2.0
    assert b[0, 0] == 2.0
    assert b[1, 1] == 8.0


def test_matmul():
    a = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = Matrix(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    c = a @ b
    assert c.shape == (2, 2)
    assert c[0, 0] == 58.0
    assert c[0, 1] == 64.0
    assert c[1, 0] == 139.0
    assert c[1, 1] == 154.0


def test_matmul_dimension_mismatch():
    a = Matrix.zeros(2, 3)
    b = Matrix.zeros(2, 3)
    with pytest.raises(ValueError):
        a @ b


def test_transpose_large_matches_sequential_reference():
    # 320x320 = 102_400 elements, over PARALLEL_TRANSPOSE_ELEMENTS (100_000),
    # so this exercises the rayon chunked path rather than the plain loop.
    rows, cols = 320, 320
    values = [float(r * cols + c) for r in range(rows) for c in range(cols)]
    m = Matrix(rows, cols, values)

    t = m.transpose()

    assert t.shape == (cols, rows)
    for r in (0, 1, rows - 1):
        for c in (0, 1, cols - 1):
            assert t[c, r] == m[r, c]


def test_matmul_large_matches_python_reference():
    # 50x50 @ 50x50 = 125_000 multiply-adds, over PARALLEL_MATMUL_FLOPS
    # (100_000), so this exercises the rayon chunked path rather than the
    # plain loop.
    n = 50
    a_vals = [[float((r + c) % 7) for c in range(n)] for r in range(n)]
    b_vals = [[float((r * 2 + c) % 5) for c in range(n)] for r in range(n)]

    a = Matrix(n, n, [v for row in a_vals for v in row])
    b = Matrix(n, n, [v for row in b_vals for v in row])

    c = a @ b

    expected = [
        [sum(a_vals[i][k] * b_vals[k][j] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]
    for i in (0, 1, n - 1):
        for j in (0, 1, n - 1):
            assert c[i, j] == pytest.approx(expected[i][j])
