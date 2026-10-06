"""Tests for the extraction of tables from the OQMD SQL dump."""

import gzip
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from oqmd_extract import InvalidDump, extract, parse_values  # noqa: E402

COLUMNS = ["id", "label", "energy", "parent_id"]
TYPES = ["int", "varchar", "double", "int"]


def test_parse_numbers_and_null():
    table = parse_values(
        "(1,'a',-1.5,NULL),(2,NULL,2e-05,7),(3,'c',NULL,-4)",
        COLUMNS,
        TYPES,
        ["id", "energy", "parent_id"],
    )

    assert table.column_names == ["id", "energy", "parent_id"]
    assert table["id"].to_pylist() == [1, 2, 3]
    assert table["energy"].to_pylist() == [-1.5, 2e-05, None]
    assert table["parent_id"].to_pylist() == [None, 7, -4]


def test_parse_strings_with_separators_and_escapes():
    values = r"(1,'a),(b',0,1),(2,'it\'s, (ok)',1,2),(3,'back\\slash',2,3),(4,'',3,4)"
    table = parse_values(values, COLUMNS, TYPES, ["id", "label"])

    assert table["id"].to_pylist() == [1, 2, 3, 4]
    assert table["label"].to_pylist() == ["a),(b", "it's, (ok)", "back\\slash", ""]


def test_parse_single_column():
    table = parse_values("(1,'a',0,1),(2,'b',1,2)", COLUMNS, TYPES, ["label"])

    assert table["label"].to_pylist() == ["a", "b"]


@pytest.mark.parametrize(
    "values",
    [
        "(1,'a',0,1),(2,'b',1",  # truncated row
        "(1,'a',0,1),(2,'b',1,2,3)",  # too many values
        "(1,'a',0,1) (2,'b',1,2)",  # missing separator
        "(1,'a',0,1),,(2,'b',1,2)",  # empty row
    ],
)
def test_parse_invalid_statement(values):
    with pytest.raises(InvalidDump):
        parse_values(values, COLUMNS, TYPES, ["id"])


def write_dump(path, tables):
    lines = ["-- MySQL dump", ""]
    for name, (columns, statements) in tables.items():
        lines.append(f"CREATE TABLE `{name}` (")
        lines += [f"  `{column}` {sql_type}," for column, sql_type in columns]
        lines += [
            "  PRIMARY KEY (`id`)",
            ") ENGINE=InnoDB;",
            f"LOCK TABLES `{name}` WRITE;",
        ]
        lines += [f"INSERT INTO `{name}` VALUES {values};" for values in statements]
        lines.append("UNLOCK TABLES;")

    with gzip.open(path, "wt", encoding="latin-1") as fp:
        fp.write("\n".join(lines) + "\n")


@pytest.fixture
def dump(tmp_path):
    path = tmp_path / "dump.sql.gz"
    write_dump(
        path,
        {
            "atoms": (
                [
                    ("id", "int(11) NOT NULL"),
                    ("element_id", "varchar(9)"),
                    ("x", "double"),
                ],
                ["(1,'Na',0),(2,'Cl',0.5)", "(3,'Fe',0.25)"],
            ),
            "ignored": (
                [("id", "int(11)"), ("text", "longtext")],
                ["(1,'INSERT INTO')"],
            ),
            "entries": (
                [("id", "int(11) NOT NULL"), ("duplicate_of_id", "int(11)")],
                ["(10,NULL),(11,10)"],
            ),
        },
    )
    return path


def test_extract(tmp_path, dump):
    counts = extract(
        dump,
        tmp_path / "tables",
        {"atoms": ["element_id", "x"], "entries": ["id", "duplicate_of_id"]},
        num_workers=2,
    )

    assert counts == {"atoms": 3, "entries": 2}

    atoms = pq.read_table(tmp_path / "tables" / "atoms.parquet")
    assert atoms["element_id"].to_pylist() == ["Na", "Cl", "Fe"]
    assert atoms["x"].to_pylist() == [0.0, 0.5, 0.25]

    entries = pq.read_table(tmp_path / "tables" / "entries.parquet")
    assert entries["duplicate_of_id"].to_pylist() == [None, 10]
    assert not (tmp_path / "tables" / "ignored.parquet").exists()


def test_extract_missing_column(tmp_path, dump):
    with pytest.raises(InvalidDump, match="occupancy"):
        extract(dump, tmp_path / "tables", {"atoms": ["occupancy"]}, num_workers=1)


def test_extract_missing_table(tmp_path, dump):
    with pytest.raises(InvalidDump, match="structures"):
        extract(dump, tmp_path / "tables", {"structures": ["id"]}, num_workers=1)


def test_parse_kept_columns_in_any_order():
    """Anti-regression: values must follow the kept names, not the schema order."""
    table = parse_values(
        "(1,'a',-1.5,10),(2,'b',2.5,20)", COLUMNS, TYPES, ["parent_id", "label", "id"]
    )

    assert table.column_names == ["parent_id", "label", "id"]
    assert table["parent_id"].to_pylist() == [10, 20]
    assert table["label"].to_pylist() == ["a", "b"]
    assert table["id"].to_pylist() == [1, 2]
