#!/usr/bin/env python3
"""Re-derive TMF8829 results from a capture log, with no hardware attached.

Reads the raw payload bytes back out of the log and runs the vendor parsers over them, so a field
capture can be re-interpreted offline with a different peak selection or threshold. Reproducing
`capture_8x8_long_range.py`'s printed grid from a log is the check that the capture is complete.

Args:
    log: path to a .ndjson.gz capture log.
    --peak: which peak per pixel to print, 0..3. Default 0.
    --set: only show this 1-based set number; default shows every set.
    --summary: print the header and per-set statistics only, no grids.
    --histograms: print the histogram bins of one zone, given as --zone Y,X.
    --zone: zone to use for --histograms, as "y,x". Default 0,0.
"""

import argparse
import base64
import sys

# capture_log puts the vendor driver on sys.path and imports only its pure-ctypes modules, so
# replaying needs neither pyzmq nor the EVM.
import capture_log
from capture_log import NO_TARGET


def print_header(record: dict) -> None:
    cfg = record.get("config", {})
    device = record.get("device", {})
    print("label      {}".format(record.get("label")))
    print("captured   {}".format(record.get("host_utc")))
    print("device     serial=0x{:08x} fw={} evm={}".format(
        device.get("deviceSerialNumber", 0), device.get("fwVersion"),
        device.get("evmVersion_ascii", "")))
    print("preconfig  {}  cfg_client={}".format(
        record.get("preconfig"), record.get("is_cfg_client")))
    if record.get("config_applied") is False:
        print("           configuration set by the device, not by the capture script")
    if record.get("config_complete") is False:
        bad = ", ".join("{} needed {} got {}".format(m["field"], m["needed"], m["actual"])
                        for m in record.get("config_shortfalls", []))
        print("           INCOMPLETE -- channels missing from this capture: {}".format(bad))
    print("config     fp_mode={} period={}ms iterations={} histograms={} nr_peaks={} select={}".format(
        cfg.get("fp_mode"), cfg.get("period"), cfg.get("iterations"),
        cfg.get("histograms"), cfg.get("nr_peaks"), cfg.get("select")))
    print("           confidence_threshold={} signal_level={} bin_shift={} peak_bins={}".format(
        cfg.get("confidence_threshold"), cfg.get("signal_level"),
        cfg.get("bin_shift"), cfg.get("peak_bins")))
    if record.get("git_sha"):
        print("git        {}".format(record["git_sha"]))
    notes = record.get("notes")
    if notes:
        complaint = capture_log.notes_complaint(notes)
        print("notes      {} ({} fields filled{})".format(
            notes.get("path"), notes.get("fields_filled", "?"),
            "; " + complaint if complaint else ""))
    else:
        print("notes      none recorded")


def reparse(record: dict, decode_histograms: bool, to_mm: bool) -> dict:
    payload = base64.b64decode(record["raw"])
    if len(payload) != record.get("raw_len", len(payload)):
        sys.stderr.write("set {}: raw_len mismatch\n".format(record.get("seq")))
    return capture_log.decode_set(payload, decode_histograms, to_mm)


def set_stats(decoded: dict, peak_index: int, unit: str) -> str:
    pixels = decoded.get("pixels")
    if not pixels:
        return "no result frames"
    distances = [
        pixel["peaks"][peak_index]["distance"]
        for row in pixels
        for pixel in row
        if pixel["peaks"][peak_index]["distance"] not in (None, NO_TARGET)
    ]
    total = sum(len(row) for row in pixels)
    if not distances:
        return "targets=0/{}".format(total)
    return "targets={}/{} min={:.0f}{u} max={:.0f}{u} mean={:.0f}{u}".format(
        len(distances), total, min(distances), max(distances),
        sum(distances) / len(distances), u=unit,
    )


def print_zone_histogram(decoded: dict, zone: tuple) -> None:
    histograms = decoded.get("histograms")
    if not histograms:
        print("  no histograms in this set (captured with --no-histograms?)")
        return
    y, x = zone
    if y >= len(histograms) or x >= len(histograms[y]):
        print("  zone {},{} outside the {}x{} grid".format(
            y, x, len(histograms), len(histograms[0])))
        return
    bins = histograms[y][x]
    if not bins:
        print("  zone {},{} has no histogram".format(y, x))
        return
    peak_bin = max(range(len(bins)), key=lambda i: bins[i])
    print("  zone {},{}: {} bins, peak at bin {} ({} hits), baseline {}".format(
        y, x, len(bins), peak_bin, bins[peak_bin], min(bins)))
    print("  " + " ".join(str(v) for v in bins))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", help="capture log to replay")
    parser.add_argument("--peak", type=int, default=0, choices=range(4), help="peak per pixel")
    parser.add_argument("--set", type=int, help="only show this 1-based set number")
    parser.add_argument("--summary", action="store_true", help="statistics only, no grids")
    parser.add_argument("--histograms", action="store_true", help="print one zone's histogram bins")
    parser.add_argument("--zone", default="0,0", help="zone for --histograms, as \"y,x\"")
    args = parser.parse_args()

    try:
        zone = tuple(int(v) for v in args.zone.split(","))
        if len(zone) != 2:
            raise ValueError
    except ValueError:
        return "--zone must be two integers, as \"y,x\""

    to_mm, unit = False, "bins"
    sets = 0
    for record in capture_log.read_log(args.log):
        if record.get("type") == "header":
            print_header(record)
            to_mm = record.get("config", {}).get("select", 0) >= 1
            unit = "mm" if to_mm else "bins"
            print()
            continue
        if record.get("type") != "set":
            continue

        sets += 1
        seq = record.get("seq", sets)
        if args.set is not None and seq != args.set:
            continue

        try:
            decoded = reparse(record, decode_histograms=args.histograms, to_mm=to_mm)
        except Exception as exc:
            print("set {}: raw payload does not parse ({}: {})".format(
                seq, type(exc).__name__, exc))
            continue
        if record.get("decoded", {}).get("error"):
            print("set {}: failed to decode during capture ({}), reparsed here".format(
                seq, record["decoded"]["error"]))

        warnings = capture_log.frame_warnings(decoded)
        header = decoded["frames"][0]["header"] if decoded.get("frames") else {}
        print("set {} {} frame={} temp={}C refPos={} bdv={} {}{}".format(
            seq, record.get("host_utc", ""), header.get("fNumber"),
            (header.get("temperature") or [None, None, None])[2], header.get("refPos"),
            header.get("bdv"), set_stats(decoded, args.peak, unit),
            "  warnings=" + warnings if warnings else "",
        ))

        if not args.summary and decoded.get("pixels"):
            capture_log.print_grid(decoded["pixels"], args.peak, unit)

        if args.histograms:
            print_zone_histogram(decoded, zone)

    if sets == 0:
        print("no measurement sets in {}".format(args.log))
    else:
        print("\n{} sets in {}".format(sets, args.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
