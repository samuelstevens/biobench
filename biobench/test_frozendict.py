import pytest
from hypothesis import given
from hypothesis import strategies as st

from .helpers import FrozenDict


def test_construct_and_access():
    fd = FrozenDict(a=1, b=2)
    assert fd["a"] == 1
    assert list(fd) == ["a", "b"]
    assert len(fd) == 2


def test_hashability():
    fd = FrozenDict(x=42)
    s = {fd}
    assert fd in s
    d = {fd: "cached"}
    assert d[fd] == "cached"


def test_equality_with_dict_and_frozendict():
    m = {"k": "v", "p": 3}
    assert FrozenDict(m) == m
    assert FrozenDict(m) == FrozenDict(m)


def test_kwargs_unpacking():
    def f(**kw):  # noqa: D401
        return kw

    fd = FrozenDict(alpha=1, beta=2)
    assert f(**fd) == {"alpha": 1, "beta": 2}


def test_iter_behaviour():
    fd = FrozenDict(a=1, b=2, c=3)
    assert set(iter(fd)) == {"a", "b", "c"}
    assert list(fd.values()) == [1, 2, 3]


def test_repr():
    fd = FrozenDict(foo="bar")
    assert "FrozenDict" in repr(fd) and "foo" in repr(fd)


# ---------- Hypothesis property tests ----------


@given(st.dictionaries(st.text(), st.integers()))
def test_roundtrip_equivalence(mapping):
    fd = FrozenDict(mapping)
    assert dict(fd) == mapping
    assert FrozenDict(mapping) == fd


@given(st.dictionaries(st.text(), st.integers()))
def test_hash_consistency(mapping):
    fd1 = FrozenDict(mapping)
    fd2 = FrozenDict(mapping)
    assert hash(fd1) == hash(fd2)


@given(st.dictionaries(st.text(), st.integers()))
def test_immutability(mapping):
    fd = FrozenDict(mapping)
    with pytest.raises(TypeError):
        fd["new"] = 5  # type: ignore[misc]


# ---------- edge-case tests ----------


def test_empty_dict():
    fd = FrozenDict()
    assert len(fd) == 0
    assert hash(fd) == hash(FrozenDict())


def test_tuple_keys():
    key = (1, 2)
    fd = FrozenDict({key: "val"})
    assert fd[key] == "val"


def test_nested_frozendict():
    inner = FrozenDict(x=1)
    outer = FrozenDict(inner=inner)
    assert outer["inner"]["x"] == 1


def test_large_dict():
    fd = FrozenDict({i: i * i for i in range(10_000)})
    assert fd[1234] == 1234 * 1234


def test_different_dicts_have_different_hash():
    assert hash(FrozenDict(a=1)) != hash(FrozenDict(b=1))


@given(st.dictionaries(st.text(), st.integers() | st.text() | st.lists(st.integers())))
def test_pickle_roundtrip(mapping):
    import pickle

    original = FrozenDict(mapping)
    serialized = pickle.dumps(original)
    deserialized = pickle.loads(serialized)

    assert original == deserialized
    assert hash(original) == hash(deserialized)


@given(st.dictionaries(st.text(), st.integers() | st.text() | st.lists(st.integers())))
def test_json_roundtrip(mapping):
    import json

    original = FrozenDict(mapping)
    serialized = json.dumps(dict(original))
    deserialized = FrozenDict(json.loads(serialized))

    assert original == deserialized
    assert hash(original) == hash(deserialized)


@given(st.dictionaries(st.text(), st.integers() | st.text() | st.lists(st.integers())))
def test_cloudpickle_roundtrip(mapping):
    pytest.importorskip("cloudpickle")
    import cloudpickle

    original = FrozenDict(mapping)
    serialized = cloudpickle.dumps(original)
    deserialized = cloudpickle.loads(serialized)

    assert original == deserialized
    assert hash(original) == hash(deserialized)


# Define a recursive dictionary strategy for nested dictionaries
@st.composite
def nested_dicts(draw, max_depth=3):
    if max_depth <= 0:
        return draw(st.dictionaries(
            st.text(), 
            st.integers() | st.text() | st.lists(st.integers())
        ))
    
    return draw(st.dictionaries(
        st.text(),
        st.integers() | st.text() | st.lists(st.integers()) | nested_dicts(max_depth - 1)
    ))


@given(nested_dicts())
def test_nested_dict_serialization(nested_mapping):
    """Test serialization roundtrips with nested dictionaries."""
    import pickle
    import json
    
    # Convert any nested dictionaries to FrozenDict
    def convert_nested(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = FrozenDict(convert_nested(v))
            else:
                result[k] = v
        return result
    
    converted = convert_nested(nested_mapping)
    original = FrozenDict(converted)
    
    # Test pickle
    pickle_serialized = pickle.dumps(original)
    pickle_deserialized = pickle.loads(pickle_serialized)
    assert original == pickle_deserialized
    assert hash(original) == hash(pickle_deserialized)
    
    # Test JSON for serializable parts
    try:
        json_serialized = json.dumps(dict(original))
        json_deserialized = FrozenDict(json.loads(json_serialized))
        assert original == json_deserialized
        assert hash(original) == hash(json_deserialized)
    except (TypeError, ValueError):
        # Skip JSON test if the dictionary contains non-serializable objects
        pass
