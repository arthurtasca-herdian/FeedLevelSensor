#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys

import numpy as np

import capture_log
from capture_log import NO_TARGET


def find_set(path: str, seq: int) -> tuple:
    header, record = None, None
    for entry in capture_log.read_log(path):
        if entry.get("type") == "header":
            header = entry
        elif entry.get("type") == "set" and entry.get("seq") == seq:
            record = entry
            break
    if header is None:
        raise SystemExit("{}: no header record; not a capture log".format(path))
    if record is None:
        raise SystemExit("{}: no set with seq={}".format(path, seq))
    if record.get("decoded", {}).get("error"):
        raise SystemExit("set {}: failed to decode during capture ({})".format(
            seq, record["decoded"]["error"]))
    return header, record


def to_grids(pixels: list, peak_index: int) -> tuple:
    rows, cols = len(pixels), len(pixels[0])
    distance = np.full((rows, cols), np.nan)
    signal = np.full((rows, cols), np.nan)
    snr = np.full((rows, cols), np.nan)

    for y, row in enumerate(pixels):
        for x, pixel in enumerate(row):
            peak = pixel["peaks"][peak_index]
            # A missing target stays NaN rather than becoming zero, so gaps read as gaps instead
            # of as a surface at the sensor.
            if peak["distance"] in (None, NO_TARGET):
                continue
            distance[y, x] = peak["distance"]
            signal[y, x] = peak["signal"] if peak["signal"] is not None else np.nan
            snr[y, x] = peak["snr"] if peak["snr"] is not None else np.nan

    return distance, signal, snr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", help="capture log to read")
    parser.add_argument("--set", type=int, required=True, help="1-based set number to export")
    parser.add_argument("--peak", type=int, default=0, choices=range(4), help="peak per zone")
    parser.add_argument("-o", "--out", help="destination .npz")
    args = parser.parse_args()

    header, record = find_set(args.log, args.set)

    config = header["config"]
    # `select` says whether the device reports distances or histogram bin indices. Exporting bins
    # as millimetres would put the surface at a plausible-looking wrong depth, so refuse instead.
    if config["select"] < 1:
        raise SystemExit(
            "{}: select={}, so peaks are histogram bin indices, not distances".format(
                args.log, config["select"]))

    decoded = capture_log.decode_set(base64.b64decode(record["raw"]), to_mm=True)
    pixels = decoded.get("pixels")
    if not pixels:
        raise SystemExit("set {}: no result frame in the payload".format(args.set))

    distance, signal, snr = to_grids(pixels, args.peak)
    warnings = capture_log.frame_warnings(decoded)

    meta = {
        "source": os.path.abspath(args.log),
        "label": header.get("label"),
        "seq": record.get("seq"),
        "host_utc": record.get("host_utc"),
        "peak": args.peak,
        "fp_mode": decoded["frames"][0]["fp_mode"],
        "select": config["select"],
        "unit": "mm",
        "integrity_ok": bool(decoded["integrity_ok"]),
        "warnings": warnings,
        "git_sha": header.get("git_sha", ""),
    }

    out = args.out
    if out is None:
        stem = os.path.basename(args.log).split(".ndjson")[0]
        out = "{}-set{:04d}.npz".format(stem, args.set)

    np.savez(out, distances_mm=distance, signal=signal, snr=snr, meta=json.dumps(meta))

    finite = distance[np.isfinite(distance)]
    print("{} -> {}".format(args.log, out))
    print("set {} {}  {}x{} zones  targets={}/{}".format(
        args.set, record.get("host_utc", ""), distance.shape[0], distance.shape[1],
        finite.size, distance.size))
    if finite.size:
        print("distance   min={:.0f}mm max={:.0f}mm mean={:.0f}mm".format(
            finite.min(), finite.max(), finite.mean()))
    if warnings:
        print("warnings   {}".format(warnings))
    if not meta["integrity_ok"]:
        print("integrity  FAILED -- container magic or frame EOF markers did not check out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
