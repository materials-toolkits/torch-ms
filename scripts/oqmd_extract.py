# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "tqdm"]
# ///
"""Extract tables of the OQMD SQL dump to parquet files, without a MySQL server.

The compressed dump is read once as a stream. Only the requested columns of
the requested tables are kept:

    uv run scripts/oqmd_extract.py qmdb__v1_8__022026.sql.gz tables/ --schema
    uv run scripts/oqmd_extract.py qmdb__v1_8__022026.sql.gz tables/
"""

import argparse
import multiprocessing as mp
import os
import re
import subprocess
from collections import deque
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tqdm

# Columns kept for each table, checked against the schema found in the dump.
TABLES: dict[str, list[str]] = {
    "atoms": ["structure_id", "element_id", "x", "y", "z", "occupancy"],
    "structures": [
        "id",
        "entry_id",
        "label",
        "natoms",
        "x1",
        "x2",
        "x3",
        "y1",
        "y2",
        "y3",
        "z1",
        "z2",
        "z3",
    ],
    "calculations": [
        "id",
        "entry_id",
        "label",
        "output_id",
        "energy_pa",
        "band_gap",
        "converged",
    ],
    "formation_energies": [
        "id",
        "entry_id",
        "calculation_id",
        "fit_id",
        "delta_e",
        "stability",
    ],
    "entries": ["id", "duplicate_of_id"],
}

STRING = r"'(?:[^'\\]|\\.)*'"
VALUE = rf"(?:NULL|{STRING}|[^,()']+)"

CREATE_TABLE = re.compile(rb"^CREATE TABLE `(\w+)` \($")
COLUMN = re.compile(rb"^\s+`(\w+)` (\w+)")
INSERT = re.compile(rb"^INSERT INTO `(\w+)` VALUES ")
ESCAPED = re.compile(r"\\(.)")

INTEGERS = {"int", "bigint", "smallint", "tinyint", "mediumint"}
FLOATS = {"double", "float", "decimal"}


class InvalidDump(Exception):
    """Raised when the dump does not have the expected format."""


def arrow_type(sql_type: str) -> pa.DataType:
    """Get the arrow type used to store a column of the dump."""
    if sql_type in INTEGERS:
        return pa.int64()
    if sql_type in FLOATS:
        return pa.float64()
    return pa.string()


def row_pattern(columns: list[str], kept: list[str]) -> re.Pattern:
    """Build the pattern matching a row, with a group per kept column."""
    values = [f"({VALUE})" if name in kept else VALUE for name in columns]
    return re.compile(r"\(" + ",".join(values) + r"\)")


def parse_values(
    values: str, columns: list[str], types: list[str], kept: list[str]
) -> pa.Table:
    """Parse the rows of an INSERT statement.

    Args:
        values: Rows of the statement, as "(...),(...)".
        columns: Name of all the columns of the table.
        types: SQL type of all the columns of the table.
        kept: Name of the columns to extract.

    Returns:
        Table with the kept columns.

    Raises:
        InvalidDump: If some text of the statement is not a row.
    """
    pattern = row_pattern(columns, kept)
    rows = pattern.findall(values)

    # every character must belong to a row or be a separator between rows
    if pattern.sub("", values) != "," * (len(rows) - 1):
        raise InvalidDump("an INSERT statement contains text that is not a row")

    if len(kept) == 1:
        rows = [(row,) for row in rows]

    # groups of the pattern follow the order of the columns of the table
    captured = [name for name in columns if name in kept]

    arrays = {}
    for name, raw in zip(captured, zip(*rows)):
        dtype = arrow_type(types[columns.index(name)])
        column = pa.array(raw, type=pa.string())
        column = pc.if_else(pc.equal(column, "NULL"), None, column)

        if pa.types.is_string(dtype):
            column = pa.array(
                [
                    None if value is None else ESCAPED.sub(r"\1", value[1:-1])
                    for value in column.to_pylist()
                ],
                type=pa.string(),
            )
        else:
            column = pc.cast(column, dtype)

        arrays[name] = column

    return pa.table([arrays[name] for name in kept], names=kept)


def _parse_statement(args: tuple[bytes, list[str], list[str], list[str]]) -> pa.Table:
    statement, columns, types, kept = args
    values = statement[statement.index(b" VALUES ") + 8 :].rstrip(b"\r\n")

    if not values.endswith(b";"):
        raise InvalidDump("an INSERT statement is not terminated")

    return parse_values(values[:-1].decode("latin-1"), columns, types, kept)


def read_dump(source: str | os.PathLike) -> Iterator[tuple[str, bytes | dict]]:
    """Iterate over the tables of a dump.

    Yields:
        ("schema", {table: [(column, type), ...]}) when a table is declared and
        (table, statement) for each of its INSERT statements.
    """
    with subprocess.Popen(
        ["gzip", "-dc", os.fspath(source)], stdout=subprocess.PIPE, bufsize=1 << 24
    ) as process:
        assert process.stdout is not None

        table = None
        for line in process.stdout:
            if line.startswith(b"INSERT INTO"):
                match = INSERT.match(line)
                if match is None:
                    raise InvalidDump("invalid INSERT statement")
                yield match.group(1).decode(), line
                continue

            match = CREATE_TABLE.match(line)
            if match is not None:
                table = (match.group(1).decode(), [])
                continue

            if table is not None:
                match = COLUMN.match(line)
                if match is not None:
                    table[1].append((match.group(1).decode(), match.group(2).decode()))
                elif line.startswith(b")"):
                    yield "schema", {table[0]: table[1]}
                    table = None

    if process.returncode != 0:
        raise InvalidDump(f"gzip failed with code {process.returncode}")


def print_schema(source: str | os.PathLike) -> None:
    """Print the columns and the size of the data of each table."""
    size: dict[str, list[int]] = {}

    with tqdm.tqdm(desc="reading", unit="B", unit_scale=True) as progress:
        for name, content in read_dump(source):
            if isinstance(content, dict):
                for table, columns in content.items():
                    size[table] = [0, 0]
                    tqdm.tqdm.write(
                        f"{table}: " + ", ".join(f"{c} {t}" for c, t in columns)
                    )
            else:
                size[name][0] += 1
                size[name][1] += len(content)
                progress.update(len(content))

    for table, (statements, num_bytes) in size.items():
        print(f"{table:40s} {statements:8d} statements {num_bytes / 1e6:12.1f} MB")


def extract(
    source: str | os.PathLike,
    destination: str | os.PathLike,
    tables: dict[str, list[str]],
    num_workers: int,
) -> dict[str, int]:
    """Extract columns of tables to "<destination>/<table>.parquet".

    Returns:
        Number of rows of each extracted table.

    Raises:
        InvalidDump: If a table or a column is missing from the dump.
    """
    os.makedirs(destination, exist_ok=True)

    schema: dict[str, tuple[list[str], list[str]]] = {}
    writers: dict[str, pq.ParquetWriter] = {}
    counts = {table: 0 for table in tables}
    pending: deque = deque()

    def write(result) -> None:
        table, parsed = result
        parsed = parsed.get()

        if table not in writers:
            writers[table] = pq.ParquetWriter(
                os.path.join(destination, f"{table}.parquet"),
                parsed.schema,
                compression="zstd",
            )

        writers[table].write_table(parsed)
        counts[table] += parsed.num_rows

    with mp.Pool(num_workers) as pool:
        with tqdm.tqdm(desc="extracting", unit="B", unit_scale=True) as progress:
            for name, content in read_dump(source):
                if isinstance(content, dict):
                    for table, columns in content.items():
                        if table not in tables:
                            continue

                        names = [column for column, _ in columns]
                        missing = [c for c in tables[table] if c not in names]
                        if len(missing) > 0:
                            raise InvalidDump(
                                f"columns {missing} not in table {table}: {names}"
                            )

                        schema[table] = (names, [t for _, t in columns])
                    continue

                progress.update(len(content))

                if name not in tables:
                    continue

                names, types = schema[name]
                args = (content, names, types, tables[name])
                pending.append((name, pool.apply_async(_parse_statement, (args,))))

                # bounded queue: statements are written in the order of the dump
                while len(pending) >= 4 * num_workers:
                    write(pending.popleft())

            while len(pending) > 0:
                write(pending.popleft())

    for writer in writers.values():
        writer.close()

    missing = [table for table in tables if table not in schema]
    if len(missing) > 0:
        raise InvalidDump(f"tables {missing} not in the dump")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("source", help="dump of the database (sql.gz format)")
    parser.add_argument("destination", help="directory of the parquet files")
    parser.add_argument(
        "--schema",
        action="store_true",
        help="only print the tables of the dump and the size of their data",
    )
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if args.schema:
        print_schema(args.source)
        return

    counts = extract(args.source, args.destination, TABLES, args.num_workers)

    for table, count in counts.items():
        print(f"{table}: {count} rows")


if __name__ == "__main__":
    main()
