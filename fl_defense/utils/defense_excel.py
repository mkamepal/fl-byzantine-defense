"""Atomic snapshots of already-computed defense results, owned by the server.

No scoring is performed here. Existing workbook rows are merged by round/client
and conflicting history is rejected, even if a writer is reconstructed.
"""

import json
import logging
import math
import os
from pathlib import Path
import tempfile

from filelock import FileLock
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from fl_defense.detection.parameters import client_number


# Header -> path in the canonical client result. Round/client/head are added by
# the exporter; measurements and decisions are copied, never recalculated.
COLUMNS = {
    "Scores": [("Clustering Score", "clustering.score"), ("Direction Score", "direction.score"),
               ("Magnitude Score", "magnitude.score"), ("Validation Score", "validation.score"),
               ("Trust Score", "trust_score"), ("Selected", "selected")],
    "Direction": [("Head Client ID", "head_client"), ("Angle to Head (degrees)", "direction.angle"),
                  ("Direction Threshold (degrees)", "direction.threshold_degrees"),
                  ("Client Residual Norm", "direction.residual_norm"),
                  ("Head Residual Norm", "direction.head_residual_norm"),
                  ("Client Zero Residual", "direction.zero_residual"),
                  ("Head Zero Residual", "direction.head_zero_residual"), ("Direction Score", "direction.score")],
    "Magnitude": [("Update L2 Norm", "magnitude.norm"), ("Head Update L2 Norm", "magnitude.head_norm"),
                  ("Relative Tolerance", "magnitude.threshold"), ("Lower Bound", "magnitude.lower_bound"),
                  ("Upper Bound", "magnitude.upper_bound"), ("Magnitude Score", "magnitude.score")],
    "Validation": [("Validation Loss", "validation.loss"), ("Validation Accuracy", "validation.accuracy"),
                   ("Validation Example Count", "validation.example_count"),
                   ("Head Validation Accuracy", "validation.head_accuracy"),
                   ("Accuracy Tolerance", "validation.accuracy_tolerance"),
                   ("Minimum Accepted Accuracy", "validation.minimum_accuracy"), ("Validation Score", "validation.score")],
    "Clustering": [("Cluster Label", "clustering.label"), ("Membership Strength", "clustering.membership_strength"),
                   ("PCA Coordinates (JSON)", "clustering.pca_coordinates"),
                   ("PCA Distance to Head", "clustering.head_distance"),
                   ("Noise Distance Threshold", "clustering.noise_distance_threshold"), ("Clustering Score", "clustering.score")],
}


def headers(sheet):
    return ["Round", "Client ID"] + [name for name, _ in COLUMNS[sheet]]


def _value(record, path):
    value = record
    for key in path.split("."):
        value = value[key]
    return value


def _validate_row(sheet, row):
    if len(row) != len(headers(sheet)) or type(row[0]) is not int or row[0] < 1:
        raise ValueError("Invalid completed-round row")
    client_number(row[1])
    for (header, _), value in zip(COLUMNS[sheet], row[2:], strict=True):
        if header == "Head Client ID":
            client_number(value)
        elif header == "PCA Coordinates (JSON)":
            coords = json.loads(value)
            if not isinstance(coords, list) or not coords or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in coords):
                raise ValueError("Invalid PCA coordinates")
        elif header in ("Selected", "Client Zero Residual", "Head Zero Residual"):
            if type(value) is not bool:
                raise ValueError(f"{header} must be Boolean")
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Missing or non-finite {header}")
            if header.endswith("Score") and header != "Trust Score":
                if type(value) is not int or value not in (0, 1):
                    raise ValueError(f"{header} must be binary")
            if header in ("Trust Score", "Validation Accuracy", "Head Validation Accuracy", "Membership Strength",
                          "Relative Tolerance", "Accuracy Tolerance") and not 0 <= value <= 1:
                raise ValueError(f"Invalid {header}")
            if header == "Validation Example Count" and (type(value) is not int or value < 1):
                raise ValueError("Trusted validation example count must be positive")
            if header == "Cluster Label" and (type(value) is not int or value < -1):
                raise ValueError("Invalid cluster label")
            if ("Norm" in header or header in ("Lower Bound", "Upper Bound", "Validation Loss", "PCA Distance to Head")) and value < 0:
                raise ValueError(f"{header} must be nonnegative")
            if "(degrees)" in header and not 0 <= value <= 180:
                raise ValueError(f"Invalid {header}")


def _validate_alignment(tables):
    keys = set(tables["Scores"])
    if any(set(table) != keys for table in tables.values()):
        raise ValueError("Workbook round/client measurements are misaligned")
    for sheet, table in tables.items():
        for row in table.values():
            _validate_row(sheet, row)
    for key, scores in tables["Scores"].items():
        for i, sheet in enumerate(("Clustering", "Direction", "Magnitude", "Validation"), start=2):
            if scores[i] != tables[sheet][key][-1]:
                raise ValueError("Workbook component scores are misaligned")
        head = tables["Direction"][key][2]
        if (key[0], head) not in keys:
            raise ValueError("Trusted head missing from completed round")
    for rnd in {key[0] for key in keys}:
        round_keys = [key for key in keys if key[0] == rnd]
        heads = {tables["Direction"][key][2] for key in round_keys}
        dimensions = {len(json.loads(tables["Clustering"][key][4])) for key in round_keys}
        counts = {tables["Validation"][key][4] for key in round_keys}
        if len(heads) != 1 or len(dimensions) != 1 or len(counts) != 1:
            raise ValueError("Round head, PCA dimensions or trusted validation counts are misaligned")
        for sheet, columns in (("Direction", (4, 6, 8)), ("Magnitude", (3, 4, 5, 6)),
                               ("Validation", (5, 6, 7)), ("Clustering", (6,))):
            if any(len({tables[sheet][key][column] for key in round_keys}) != 1 for column in columns):
                raise ValueError(f"Round reference measurements are misaligned in {sheet}")


def round_tables(result):
    rnd, head, clients = result["round"], result["head_client"], result["clients"]
    if not clients or head not in clients:
        raise ValueError("Completed round is empty or missing its trusted head")
    tables = {sheet: {} for sheet in COLUMNS}
    for cid in sorted(clients, key=client_number):
        record = {**clients[cid], "head_client": head}
        for sheet, columns in COLUMNS.items():
            values = []
            for _, path in columns:
                value = _value(record, path)
                if path == "clustering.pca_coordinates":
                    value = json.dumps(value, allow_nan=False)
                values.append(value)
            tables[sheet][rnd, cid] = (rnd, cid, *values)
    _validate_alignment(tables)
    return tables


class DefenseExcel:
    def __init__(self, run_dir, metadata):
        self.run_dir = Path(run_dir).resolve()
        if not self.run_dir.is_dir():
            raise ValueError("Excel output must belong to an existing current run directory")
        self.path = self.run_dir / "defense_scores.xlsx"
        self.metadata = {**metadata, "Workbook semantics": "Snapshot only; editing configuration does not recalculate defenses",
                         "Round-numbering convention": "1-based completed federated rounds; initial round 0 excluded"}
        # Flower IDs can span uint64; numeric Excel cells round away digits.
        # Normalize before both writing and checking the existing configuration.
        if self.metadata.get("Run ID") is not None:
            self.metadata["Run ID"] = str(self.metadata["Run ID"])

    def _configuration_rows(self):
        return [(key, json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list))
                 else "unavailable" if value is None else value) for key, value in self.metadata.items()]

    def _read_existing(self):
        tables = {sheet: {} for sheet in COLUMNS}
        if not self.path.exists():
            return tables
        book = load_workbook(self.path, read_only=True, data_only=False)
        try:
            if book.sheetnames != [*COLUMNS, "Configuration"]:
                raise ValueError("Existing defense workbook has unexpected sheets")
            config_rows = list(book["Configuration"].values)
            if config_rows != [("Setting", "Value"), *self._configuration_rows()]:
                raise ValueError("Existing workbook belongs to a different effective run configuration")
            for sheet in COLUMNS:
                rows = iter(book[sheet].values)
                if list(next(rows)) != headers(sheet):
                    raise ValueError(f"Existing {sheet} headers differ")
                for row in rows:
                    key = row[:2]
                    if key in tables[sheet]:
                        raise ValueError("Duplicate existing round/client rows")
                    tables[sheet][key] = row
            _validate_alignment(tables)
        finally:
            book.close()
        return tables

    @staticmethod
    def _style(sheet):
        sheet.freeze_panes = "C2" if sheet.title != "Configuration" else "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 44
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            width = min(32, max(16, len(str(cell.value)) + 2))
            sheet.column_dimensions[get_column_letter(cell.column)].width = width
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = "s"  # metadata is literal, never an Excel formula
                header = sheet.cell(1, cell.column).value
                if type(cell.value) in (int, float) and "Accuracy" in header:
                    cell.number_format = "0.0000%"
                elif type(cell.value) in (int, float) and ("Norm" in header or header in ("Lower Bound", "Upper Bound")):
                    cell.number_format = "0.0000000000E+00"
                elif type(cell.value) is float and 0 < abs(cell.value) < 1e-6:
                    cell.number_format = "0.0000000000E+00"
                elif type(cell.value) is float:
                    cell.number_format = "0.0000000000"
        if sheet.title == "Clustering":
            sheet.column_dimensions["E"].width = 52
            for row in sheet.iter_rows(min_row=2):
                row[4].alignment = Alignment(wrap_text=True, vertical="top")
                sheet.row_dimensions[row[0].row].height = 16 * (1 + len(str(row[4].value)) // 48)
        if sheet.title == "Configuration":
            sheet.column_dimensions["A"].width = 40
            sheet.column_dimensions["B"].width = 100
            for row in sheet.iter_rows(min_row=2):
                row[1].alignment = Alignment(wrap_text=True, vertical="top")
                sheet.row_dimensions[row[0].row].height = min(240, 16 * (1 + len(str(row[1].value)) // 90))

    def save(self, completed_rounds):
        """Idempotent merge; fail closed and retain the last successful workbook."""
        temporary = None
        try:
            incoming = {sheet: {} for sheet in COLUMNS}
            for rnd, result in completed_rounds.items():
                if rnd != result["round"]:
                    raise ValueError("Completed-round key mismatch")
                if ("Trusted head ID" in self.metadata and result["head_client"] != self.metadata["Trusted head ID"]):
                    raise ValueError("Completed-round trusted head differs from effective configuration")
                if ("Validation-data source" in self.metadata and
                        result["validation_source"] != self.metadata["Validation-data source"]):
                    raise ValueError("Completed-round validation source differs from effective configuration")
                for sheet, table in round_tables(result).items():
                    incoming[sheet].update(table)
            with FileLock(str(self.run_dir / ".defense_scores.lock")):
                tables = self._read_existing()
                for sheet, table in incoming.items():
                    for key, row in table.items():
                        # Excel may round the final float bit on serialization.
                        if key in tables[sheet] and not _same_row(tables[sheet][key], row):
                            raise ValueError(f"Conflicting completed history: {key}")
                        tables[sheet][key] = row
                _validate_alignment(tables)
                book = Workbook()
                book.remove(book.active)
                try:
                    for sheet, table in tables.items():
                        page = book.create_sheet(sheet)
                        page.append(headers(sheet))
                        for key in sorted(table, key=lambda k: (k[0], client_number(k[1]))):
                            page.append(table[key])
                        self._style(page)
                    page = book.create_sheet("Configuration")
                    page.append(["Setting", "Value"])
                    for row in self._configuration_rows():
                        page.append(row)
                    self._style(page)
                    with tempfile.NamedTemporaryFile(dir=self.run_dir, prefix=".defense_scores_", suffix=".xlsx",
                                                     delete=False) as stream:
                        temporary = Path(stream.name)
                    book.save(temporary)
                    os.replace(temporary, self.path)
                finally:
                    book.close()
        except Exception as error:
            logging.getLogger("flwr").error("Failed to save defense workbook %s: %s", self.path, error)
            raise RuntimeError(f"Failed to save defense workbook {self.path}: {error}") from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logging.getLogger("flwr").warning("Could not remove temporary workbook %s: %s", temporary, cleanup_error)
        logging.getLogger("flwr").info("Saved completed defense results: %s", self.path)


def _same_row(left, right):
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-14, abs_tol=0) if type(a) in (int, float) and type(b) in (int, float)
        else a == b for a, b in zip(left, right, strict=True))
