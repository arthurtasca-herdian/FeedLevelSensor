"""Streaming capture log for TMF8829 field measurements.

One gzipped NDJSON file per capture: a `header` record describing the device, the full
configuration and the site notes, followed by one `set` record per measurement set.

Every `set` record carries the verbatim ZeroMQ payload. `getFramesFromMeasurementResult`,
`getFullPixelResult` and `getAllHistogramResults` are pure functions over those bytes, so the raw
field data can be re-parsed offline with a different peak selection, threshold or driver version.
The decoded values alongside it are a convenience, not the record of truth.
"""

import base64
import ctypes
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

_DRIVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver", "tmf8829")
if not os.path.isdir(_DRIVER_DIR):
    sys.exit("Driver not found at {}. Unzip TMF8829_Driver_Python_v*.zip into ./driver".format(_DRIVER_DIR))
# The vendor modules do a bare `import __init__` that must resolve to driver/tmf8829/__init__.py.
# Everything imported below is pure ctypes, so replaying a log needs no pyzmq and no hardware.
if _DRIVER_DIR not in sys.path:
    sys.path.insert(0, _DRIVER_DIR)

from tmf8829_application_common import Tmf8829AppCommon
from tmf8829_application_defines import (
    TMF8829_FID_HISTOGRAMS,
    TMF8829_FID_MASK,
    TMF8829_FID_REF_SPAD_SCAN,
    TMF8829_FID_RESULTS,
    TMF8829_FPM_MASK,
    TMF8829_FRAME_EOF,
    TMF8829_FRAME_VALID,
    TMF8829_FRAME_WARNING_HV_CP_OVERLOAD,
    TMF8829_FRAME_WARNING_VCDRV_BURST_EXCEEDED,
    TMF8829_FRAME_WARNING_VCDRV_OVERLOAD,
    struct__tmf8829FrameFooter,
    struct__tmf8829FrameHeader,
    struct__tmf8829RefSpadFrame,
    tmf8829FrameFooter,
    tmf8829FrameHeader,
    tmf8829RefSpadFrame,
)
from zeromq.tmf8829_host_com_reg import (
    TMF8829_ZEROMQ_PROTOCOL_MAGIC_NUMBER,
    tmf8829ContainerFrameHeader,
)

SCHEMA = "feedlevelsensor.capture/1"

NO_TARGET = 0

WARNING_NAMES = (
    (TMF8829_FRAME_WARNING_HV_CP_OVERLOAD, "hv_cp_overload"),
    (TMF8829_FRAME_WARNING_VCDRV_OVERLOAD, "vcdrv_overload"),
    (TMF8829_FRAME_WARNING_VCDRV_BURST_EXCEEDED, "vcdrv_burst_exceeded"),
)

_HEADER_SIZE = ctypes.sizeof(struct__tmf8829FrameHeader)
_FOOTER_SIZE = ctypes.sizeof(struct__tmf8829FrameFooter)
_PRE = Tmf8829AppCommon.PRE_HEADER_SIZE

_FID_NAMES = {
    TMF8829_FID_RESULTS: "results",
    TMF8829_FID_HISTOGRAMS: "histograms",
    TMF8829_FID_REF_SPAD_SCAN: "ref_spad_scan",
}


def _struct_to_dict(struct) -> dict:
    out = {}
    for field in struct._fields_:
        name = field[0]
        value = getattr(struct, name)
        out[name] = list(value) if hasattr(value, "__len__") else value
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


NOTES_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "configs", "field_notes.template.yaml")

_PLACEHOLDER_LABEL = "silo3-tilted20-halffull"


def read_notes(path: str) -> dict:
    # Stored verbatim rather than parsed: the notes only need to survive the trip, and embedding
    # the text keeps this script free of a YAML dependency.
    with open(path, "rb") as handle:
        raw = handle.read()
    text = raw.decode("utf-8", errors="replace")

    # An unfilled template logged as though it were measured is worse than no notes at all, so
    # count the keys still left blank and let the caller say so before the capture starts.
    blank = 0
    filled = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        if stripped.split(":", 1)[1].strip():
            filled += 1
        else:
            blank += 1

    return {
        "path": os.path.abspath(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": text,
        "fields_blank": blank,
        "fields_filled": filled,
        "placeholder_label": _PLACEHOLDER_LABEL in text,
        "is_template": os.path.abspath(path) == NOTES_TEMPLATE,
    }


def notes_complaint(notes: dict) -> str:
    # Reads notes dicts straight out of a log file, which may predate any of these keys.
    if notes.get("is_template"):
        return "this is the template itself, not a filled-in copy"
    if notes.get("placeholder_label"):
        return "still contains the template's placeholder label"
    blank = notes.get("fields_blank", 0)
    total = blank + notes.get("fields_filled", 0)
    if total and blank > total - blank:
        return "{} of {} fields still blank".format(blank, total)
    return ""


def device_info_to_dict(device) -> dict:
    info = _struct_to_dict(device)
    info["evmVersion_ascii"] = bytes(device.evmVersion).decode("ascii", errors="replace").strip()
    return info


def frame_metadata(frame: bytearray) -> dict:
    header = tmf8829FrameHeader.from_buffer_copy(bytes(frame[_PRE:_PRE + _HEADER_SIZE]))
    footer = tmf8829FrameFooter.from_buffer_copy(bytes(frame[-_FOOTER_SIZE:]))
    meta = {
        "preheader": {
            "fifostatus": frame[0],
            "systick": int.from_bytes(bytes(frame[1:5]), "little"),
        },
        "header": _struct_to_dict(header),
        "footer": _struct_to_dict(footer),
        "kind": _FID_NAMES.get(header.id & TMF8829_FID_MASK, "unknown"),
        "fp_mode": header.id & TMF8829_FPM_MASK,
        # The vendor parser does not check the markers, so corrupt bytes decode into plausible
        # looking nonsense. These two flags are what distinguishes a real frame from noise.
        "eof_ok": footer.eof == TMF8829_FRAME_EOF,
    }
    return meta


def ref_spad_to_dict(frame: bytearray) -> dict:
    size = ctypes.sizeof(struct__tmf8829RefSpadFrame)
    parsed = tmf8829RefSpadFrame.from_buffer_copy(bytes(frame[_PRE:_PRE + size]))
    return {"sum": [list(cycle) for cycle in parsed.sum]}


def decode_set(payload: bytes, decode_histograms: bool = False) -> dict:
    container_size = ctypes.sizeof(tmf8829ContainerFrameHeader)
    container = tmf8829ContainerFrameHeader.from_buffer_copy(payload[:container_size])
    body = payload[container_size:]

    result_frames, histo_frames, ref_frames = Tmf8829AppCommon.getFramesFromMeasurementResult(body)

    decoded = {
        "container": {
            "magicNumber": container.magicNumber,
            "protocolVersion": container.protocolVersion,
            "payload": container.payload,
            "hostType": container.hostType,
            "correctionFactor": container.correctionFactor,
            "deviceSerialNumber": container.deviceSerialNumber,
        },
        "frames": [frame_metadata(f) for f in list(result_frames) + list(histo_frames) + list(ref_frames)],
        "counts": {
            "result": len(result_frames),
            "histogram": len(histo_frames),
            "ref_spad": len(ref_frames),
        },
        "magic_ok": container.magicNumber == TMF8829_ZEROMQ_PROTOCOL_MAGIC_NUMBER,
    }
    decoded["integrity_ok"] = decoded["magic_ok"] and all(
        f["eof_ok"] for f in decoded["frames"]) and bool(decoded["frames"])

    if result_frames:
        # deleteNone=False keeps all four peak slots so a peak index means the same thing in every
        # record; toMM=False keeps device units, which `select` in the header record defines.
        decoded["pixels"] = Tmf8829AppCommon.getFullPixelResult(
            frames=result_frames, toMM=False, deleteNone=False, pointCloud=False, distanceToXYZ=False
        )

    if ref_frames:
        decoded["ref_spad"] = [ref_spad_to_dict(f) for f in ref_frames]

    # The bins are already in `raw`, and inlining them as JSON integers costs more than the payload
    # itself, so this stays opt-in.
    if decode_histograms and histo_frames:
        ref_histograms, mp_histograms = Tmf8829AppCommon.getAllHistogramResults(histo_frames)
        decoded["ref_histograms"] = [list(h.bin) for h in ref_histograms]
        decoded["histograms"] = [
            [list(pixel.bin) if pixel else None for pixel in row] for row in mp_histograms
        ]

    return decoded


class CaptureWriter:
    """Append-only NDJSON writer. One line per measurement set, flushed as it goes.

    Args:
        path: destination `.ndjson.gz` file.
        decode_histograms: also inline decoded histogram bins in each record.
    """

    def __init__(self, path: str, decode_histograms: bool = False):
        self.path = path
        self.decode_histograms = decode_histograms
        self.count = 0
        self.bytes_raw = 0
        self.decode_errors = 0
        self._handle = gzip.open(path, "wt", encoding="utf-8")

    def write_header(self, **fields) -> None:
        record = {
            "type": "header",
            "schema": SCHEMA,
            "host_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "git_sha": _git_sha(),
        }
        record.update(fields)
        self._write(record)

    def write_set(self, payload: bytes) -> dict:
        self.count += 1
        self.bytes_raw += len(payload)
        record = {
            "type": "set",
            "seq": self.count,
            "host_utc": time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + "{:03d}Z".format(
                int(time.time() * 1000) % 1000
            ),
            "raw_len": len(payload),
            "raw": base64.b64encode(payload).decode("ascii"),
        }
        # The raw bytes are the record of truth, so a frame the parsers choke on must still reach
        # the file: it can be diagnosed offline, and one bad set must not end a field capture.
        try:
            decoded = decode_set(payload, self.decode_histograms)
        except Exception as exc:
            self.decode_errors += 1
            decoded = {"error": "{}: {}".format(type(exc).__name__, exc)}
        record["decoded"] = decoded
        self._write(record)
        return decoded

    def _write(self, record: dict) -> None:
        self._handle.write(json.dumps(record, separators=(",", ":")))
        self._handle.write("\n")
        # A field capture can end with a pulled cable; an unflushed buffer would lose more than
        # the last line.
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def describe_warnings(status: int) -> str:
    names = [name for bit, name in WARNING_NAMES if status & bit]
    leftover = status & ~TMF8829_FRAME_VALID
    for bit, _ in WARNING_NAMES:
        leftover &= ~bit
    if leftover:
        names.append("0x{:x}".format(leftover))
    return ",".join(names)


def frame_warnings(decoded: dict) -> str:
    out = []
    if not decoded.get("magic_ok", True):
        out.append("bad-container-magic")
    for frame in decoded.get("frames", []):
        status = frame["footer"]["frameStatus"]
        if status & ~TMF8829_FRAME_VALID:
            out.append("{}:{}".format(frame["kind"], describe_warnings(status)))
        if not frame.get("eof_ok", True):
            out.append("{}:bad-eof-marker".format(frame["kind"]))
    return " ".join(out)


def scale_grid(pixels: list, to_mm: bool) -> list:
    if not to_mm:
        return pixels
    return [
        [
            {"peaks": [
                {"distance": None if p["distance"] is None else p["distance"] / 4.0}
                for p in pixel["peaks"]
            ]}
            for pixel in row
        ]
        for row in pixels
    ]


def print_grid(pixel_results: list, peak_index: int, unit: str) -> None:
    cols = len(pixel_results[0])
    print("       " + "".join("{:>8}".format("x" + str(x)) for x in range(cols)))
    for y, row in enumerate(pixel_results):
        cells = []
        for pixel in row:
            distance = pixel["peaks"][peak_index]["distance"]
            if distance is None or distance == NO_TARGET:
                cells.append("{:>8}".format("--"))
            else:
                cells.append("{:>8.0f}".format(distance))
        print("y{:<6}".format(y) + "".join(cells))

    valid = [
        pixel["peaks"][peak_index]["distance"]
        for row in pixel_results
        for pixel in row
        if pixel["peaks"][peak_index]["distance"] not in (None, NO_TARGET)
    ]
    if valid:
        print(
            "  targets={}/{}  min={:.0f}{u}  max={:.0f}{u}  mean={:.0f}{u}".format(
                len(valid), len(pixel_results) * cols, min(valid), max(valid),
                sum(valid) / len(valid), u=unit,
            )
        )
    else:
        print("  targets=0/{} (no peaks detected)".format(len(pixel_results) * cols))


def read_log(path: str):
    """Yield every record from a capture log, header first.

    A capture cut short by a pulled cable leaves a truncated gzip stream, so both an unparsable
    final line and an unterminated stream are treated as end-of-file rather than errors.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        line_number = 0
        while True:
            try:
                line = handle.readline()
            except (EOFError, OSError) as exc:
                sys.stderr.write("{}: truncated after {} lines ({})\n".format(
                    path, line_number, type(exc).__name__))
                return
            if not line:
                return
            line_number += 1
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                sys.stderr.write("{}: ignoring unparsable line {}\n".format(path, line_number))


def log_path(log_dir: str, label: str, serial: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in label).strip("-")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return os.path.join(log_dir, "{}_UID{:08x}_{}.ndjson.gz".format(safe, serial, stamp))
