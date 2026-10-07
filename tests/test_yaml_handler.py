"""src.yaml_handler: map schema validation and file reading (spec 007 FR-9).

The schema mirrors the one map_factory passes, trimmed to one key per type kind.
"""

import pytest

from src.yaml_handler import YAMLMapReader, get_optional_group_keys

SCHEMA = {
    "mandatory": {"FEM": str, "x_pitch": (int, float), "channels": int, "sum_rows_cols": bool},
    "optional_groups": [
        {"group": ["channels_j1", "channels_j2"], "type": list},
        {"group": ["Time", "Energy"], "type": list},
    ],
}
VALID = {"FEM": "FEM256", "x_pitch": 3.2, "channels": 256, "sum_rows_cols": True,
         "Time": [0, 1], "Energy": [2, 3]}


def without(key):
    return {k: v for k, v in VALID.items() if k != key}


def validate(yaml_map):
    return YAMLMapReader(SCHEMA).validate_yaml_map(yaml_map)


@pytest.mark.fr("007-FR-9")
def test_valid_map_accepted():
    assert validate(VALID) is True
    assert validate({**VALID, "x_pitch": 3}) is True


@pytest.mark.fr("007-FR-9")
def test_missing_mandatory_key_named():
    with pytest.raises(RuntimeError, match="Missing mandatory key in YAML map: channels"):
        validate(without("channels"))


@pytest.mark.fr("007-FR-9")
def test_wrong_type_named():
    with pytest.raises(RuntimeError, match="Incorrect type for mandatory key channels: expected int, got str"):
        validate({**VALID, "channels": "256"})


@pytest.mark.fr("007-FR-9", "bug-yaml-tuple-type-message")
@pytest.mark.xfail(strict=True, reason="bug-yaml-tuple-type-message: (int, float) has no __name__ -> AttributeError")
def test_wrong_type_named_for_multi_type_key():
    with pytest.raises(RuntimeError, match="Incorrect type for mandatory key x_pitch"):
        validate({**VALID, "x_pitch": "3.2"})


@pytest.mark.fr("007-FR-9", "bug-yaml-bool-as-int")
@pytest.mark.xfail(strict=True, reason="bug-yaml-bool-as-int: bool passes isinstance(value, int)")
def test_bool_rejected_for_integer_key():
    with pytest.raises(RuntimeError, match="Incorrect type for mandatory key channels"):
        validate({**VALID, "channels": True})


@pytest.mark.fr("007-FR-9")
def test_no_optional_group_rejected():
    with pytest.raises(RuntimeError, match="Exactly one of the optional groups"):
        validate(without("Energy"))
    with pytest.raises(RuntimeError, match="Exactly one of the optional groups"):
        validate({**VALID, "Energy": {"a": 1}})  # wrong type counts as absent


@pytest.mark.fr("007-FR-9")
def test_both_optional_groups_rejected():
    with pytest.raises(RuntimeError, match="More than one of the optional groups"):
        validate({**VALID, "channels_j1": [0], "channels_j2": [1]})


@pytest.mark.fr("007-FR-9")
def test_read_yaml_file(tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text("FEM: FEM256\nx_pitch: 3.2\nchannels: 256\nsum_rows_cols: true\n"
                    "channels_j1: [0, 1]\nchannels_j2: [2, 3]\n", encoding="utf-8")
    loaded = YAMLMapReader(SCHEMA).read_yaml_file(str(path))
    assert loaded["channels_j2"] == [2, 3]


@pytest.mark.fr("007-FR-9")
def test_unreadable_yaml(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("FEM: [FEM256\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Mapping file not readable"):
        YAMLMapReader(SCHEMA).read_yaml_file(str(path))


@pytest.mark.fr("007-FR-9")
def test_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="Mapping file not found"):
        YAMLMapReader(SCHEMA).read_yaml_file(str(tmp_path / "absent.yaml"))


@pytest.mark.fr("007-FR-9")
def test_optional_group_keys():
    assert get_optional_group_keys({"channels_j1": [], "channels_j2": []}) == ("channels_j1", "channels_j2")
    assert get_optional_group_keys({"Time": [], "Energy": []}) == ("Time", "Energy")
    with pytest.raises(RuntimeError, match="No valid optional group found"):
        get_optional_group_keys({"Time": []})
