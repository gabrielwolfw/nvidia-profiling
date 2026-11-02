"""Tests for reading the collector's binary kernel journal."""

import csv
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# pylint: disable-next=wrong-import-position
from energy_attribution import kernel_journal


def encode_journal(pid, names, runs):
    """Encodes a journal following the documented layout, independently."""
    data = bytearray(b"NVPMKJ01" + struct.pack("<Ii", 1, pid))
    for name_id, name in names.items():
        encoded = name.encode("utf-8")
        data += b"N" + struct.pack("<II", name_id, len(encoded)) + encoded
    for run in runs:
        data += b"R" + struct.pack("<I", len(run))
        for record in run:
            data += struct.pack("<QQIIIII", *record)
    return bytes(data)


class KernelJournalTest(unittest.TestCase):
    """Reads journals and writes the analyzer CSV."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, data):
        path = self.root / "kernel_activity.journal"
        path.write_bytes(data)
        return path

    def test_write_kernel_csv_sorts_runs_and_quotes_names(self):
        path = self.write(
            encode_journal(
                4242,
                {0: "gemm", 1: 'say "hi"'},
                [
                    [(300, 350, 0, 1, 7, 3, 0), (100, 180, 0, 1, 7, 1, 1)],
                    [(200, 260, 0, 1, 8, 2, 0), (100, 120, 0, 1, 8, 4, 0)],
                ],
            )
        )
        journal = kernel_journal.read_kernel_journal(path)
        self.assertFalse(journal.truncated)
        output = self.root / "kernel_activity.csv"

        rows = kernel_journal.write_kernel_csv(journal, output)

        self.assertEqual(rows, 4)
        lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], ",".join(kernel_journal.KERNEL_CSV_COLUMNS))
        self.assertEqual(lines[1], '0,4242,0,1,7,1,100,180,80,"say ""hi"""')
        with output.open(newline="", encoding="utf-8") as source:
            table = list(csv.DictReader(source))
        self.assertEqual(
            [row["correlation_id"] for row in table], ["1", "4", "2", "3"]
        )
        self.assertEqual(table[3]["kernel_name"], "gemm")

    def test_read_kernel_journal_skips_a_truncated_tail(self):
        complete = encode_journal(1, {0: "k"}, [[(10, 20, 0, 0, 0, 0, 0)]])
        partial_run = b"R" + struct.pack("<I", 5) + b"\x00" * 40
        journal = kernel_journal.read_kernel_journal(
            self.write(complete + partial_run)
        )
        self.assertTrue(journal.truncated)
        self.assertEqual(len(journal.records), 1)
        self.assertEqual(journal.names, {0: "k"})

    def test_read_kernel_journal_rejects_other_files(self):
        with self.assertRaises(ValueError):
            kernel_journal.read_kernel_journal(self.write(b"event_id,pid\n"))

    def test_read_kernel_journal_accepts_an_empty_journal(self):
        journal = kernel_journal.read_kernel_journal(
            self.write(encode_journal(9, {}, []))
        )
        self.assertEqual(journal.pid, 9)
        self.assertEqual(len(journal.records), 0)


if __name__ == "__main__":
    unittest.main()
