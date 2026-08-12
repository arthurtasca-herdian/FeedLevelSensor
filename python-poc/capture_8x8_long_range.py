#!/usr/bin/env python3
"""Configure a TMF8829 EVM shield for 8x8 long range, print parsed results and log them for offline analysis.

The shield-board transport in the vendor driver (aos_com.h5_com -> corefw_c) ships
Windows-only binaries, so this runs as a ZeroMQ client and talks to the vendor's
ZMQ server (tmf8829_zeromq_server.py, or the TMF8829_Driver_ZMQ_Server_Client_EXE
build) on the machine the EVM is cabled to.

Args:
    --label: what the sensor is pointed at, e.g. "silo3-tilted20-halffull". Required, and part of
        the log filename; this is the only thing that identifies a capture back at the office.
    --host: address of the machine running the TMF8829 ZeroMQ server. Default 127.0.0.1.
    --log-dir: directory for the capture log. Default ./captures. Use --no-log to only print.
    --notes: path to a filled-in field notes file, stored verbatim in the log header.
    --interval: seconds between measurements; also programmed as the device measurement period.
        Default 0.2 (5 fps).
    --count: number of frame sets to capture, 0 runs until Ctrl-C. Default 0.
    --peak: which peak per pixel to print, 0..3. Default 0. All four are always logged.
    --no-histograms: do not ask the device for raw histograms. Histograms are what let peak
        detection be re-derived offline, but the driver documents them as blocking mode, so drop
        them if the achieved frame rate is too low.
    --decode-histograms: also inline decoded histogram bins in each record. The bins are already
        in the raw payload, so this only costs space.
    --iterations: override kilo-iterations, for capturing one scene at several integration times.
    --print-every: only print a grid every Nth set, to keep a fast capture readable. Default 1.
"""

import argparse
import base64
import os
import shutil
import sys
import time

_DRIVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver", "tmf8829")
if not os.path.isdir(_DRIVER_DIR):
    sys.exit("Driver not found at {}. Unzip TMF8829_Driver_Python_v*.zip into ./driver".format(_DRIVER_DIR))
# Ahead of everything else: the vendor modules do a bare `import __init__` that must
# resolve to driver/tmf8829/__init__.py, and `aos_com` must resolve to the driver's
# local ctypes2Dict stub rather than the win_amd64 aos_com wheel.
sys.path.insert(0, _DRIVER_DIR)

from register_page_converter import RegisterPageConverter as RegConv
from tmf8829_application_registers import Tmf8829_application_registers as Tmf8829AppRegs
from tmf8829_config_page import Tmf8829_config_page as Tmf8829ConfigRegs
from zeromq import tmf8829_zeromq_client
from zeromq.tmf8829_zeromq_client import ZeroMqClient

import capture_log

PRE_CONFIG_NAME = "CMD_LOAD_CFG_8X8_LONG_RANGE"
PRE_CONFIG = Tmf8829AppRegs.TMF8829_CMD_STAT._cmd_stat._CMD_LOAD_CFG_8X8_LONG_RANGE
PRE_CONFIG_FP_MODE = 0

# Everything else stays at whatever the 8x8 long range pre-configuration page loaded. The result
# format is widened because a silo return is ambiguous: dust, the roof hatch, the far wall and the
# feed can all answer in one zone, so peak 0 is not necessarily the feed. publish=1 adds the
# reference-SPAD frame.
MEASURE_CFG = {
    "nr_peaks": 4,
    "signal_strength": 1,
    "noise_strength": 1,
    "xtalk": 1,
    "publish": 1,
}

def connect(client: ZeroMqClient, host: str) -> None:
    # connect_local() reads these module globals, and they are pinned to 127.0.0.1.
    # Loopback already reaches the Windows host under WSL2 mirrored networking;
    # redirecting them keeps --host usable for a server on another machine.
    tmf8829_zeromq_client.TMF8829_ZEROMQ_CMD_SERVER_ADDR = "tcp://{}:5557".format(host)
    tmf8829_zeromq_client.TMF8829_ZEROMQ_RESULT_SERVER_ADDR = "tcp://{}:5558".format(host)
    client.connect_local()


def configure(client: ZeroMqClient, period_ms: int, histograms: int, iterations: int = None) -> dict:
    # Both calls return False when this client does not own the configuration. Ignoring them is how
    # a capture ends up silently following someone else's setup.
    pre_ok = bool(client.set_pre_config(cmd=bytes([PRE_CONFIG])))

    baseline = RegConv.readPageToDict(client.get_config(), Tmf8829ConfigRegs())
    requested = dict(baseline)
    requested.update(MEASURE_CFG)
    requested["period"] = period_ms
    requested["histograms"] = histograms
    if iterations is not None:
        requested["iterations"] = iterations
    set_ok = bool(client.set_config(RegConv.readDictToPage(requested, Tmf8829ConfigRegs())))

    raw_back = client.get_config()
    actual = RegConv.readPageToDict(raw_back, Tmf8829ConfigRegs())
    return {
        "pre_ok": pre_ok,
        "set_ok": set_ok,
        "baseline": baseline,
        "requested": requested,
        "actual": actual,
        "raw": raw_back,
    }


def config_shortfalls(actual: dict, histograms: int) -> list:
    """Channels this capture needs but the device is not producing.

    Deliberately about capabilities, not authorship: an EVM that configures itself at boot is a
    perfectly good field setup, so what matters is whether the data contains what the offline
    analysis needs, not whether this client is the one that asked for it.
    """
    required = {"fp_mode": PRE_CONFIG_FP_MODE, "publish": 1}
    required.update(MEASURE_CFG)
    if histograms:
        required["histograms"] = 1
    return [(k, v, actual.get(k)) for k, v in sorted(required.items()) if actual.get(k) != v]


def config_preferences(requested: dict, actual: dict) -> list:
    # Timing differences change the capture's pace but not what a frame contains, so they are
    # reported and not treated as a reason to refuse.
    return [(k, requested[k], actual.get(k)) for k in ("period", "iterations")
            if actual.get(k) != requested.get(k)]


def describe_lost_channels(cfg: dict) -> list:
    lost = []
    if cfg.get("nr_peaks", 0) < 4:
        lost.append("only {} of 4 peaks per zone (multi-return and ghost detection)".format(
            cfg.get("nr_peaks")))
    if not cfg.get("histograms"):
        lost.append("no raw histograms (cannot re-derive peak detection offline)")
    if not cfg.get("publish"):
        lost.append("no reference-SPAD frame")
    for field, what in (("signal_strength", "per-peak signal"),
                        ("noise_strength", "per-zone noise"),
                        ("xtalk", "per-zone crosstalk")):
        if not cfg.get(field):
            lost.append("no {}".format(what))
    return lost


def print_set(decoded: dict, to_mm: bool, peak_index: int, unit: str) -> None:
    if decoded.get("error"):
        print("{} unparsable frame set: {} (raw bytes still logged)".format(
            time.strftime("%H:%M:%S"), decoded["error"]))
        return

    frames = decoded.get("frames", [])
    if not frames:
        print("frame set contained no frames")
        return

    first = frames[0]["header"]
    warnings = capture_log.frame_warnings(decoded)
    print("{} frame={} temperature={}C{}".format(
        time.strftime("%H:%M:%S"), first["fNumber"], first["temperature"][2],
        "  warnings=" + warnings if warnings else "",
    ))

    pixels = decoded.get("pixels")
    if not pixels:
        print("  frame set contained no result frames")
        return

    capture_log.print_grid(capture_log.scale_grid(pixels, to_mm), peak_index, unit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label",
                        help="what the sensor is pointed at; identifies the capture log. Required "
                             "unless --new-notes is given")
    parser.add_argument("--host", default="127.0.0.1", help="TMF8829 ZeroMQ server address")
    parser.add_argument("--log-dir", default="captures", help="directory for the capture log")
    parser.add_argument("--no-log", action="store_true", help="print only, write no log")
    parser.add_argument("--notes", help="filled-in copy of configs/field_notes.template.yaml, "
                                       "stored verbatim in the log header")
    parser.add_argument("--new-notes", metavar="PATH",
                        help="copy the field notes template to PATH and exit")
    parser.add_argument("--interval", type=float, default=0.2, help="seconds between measurements")
    parser.add_argument("--count", type=int, default=0, help="frame sets to capture, 0 = endless")
    parser.add_argument("--peak", type=int, default=0, choices=range(4), help="peak per pixel to print")
    parser.add_argument("--no-histograms", action="store_true", help="do not request raw histograms")
    parser.add_argument("--decode-histograms", action="store_true",
                        help="also inline decoded histogram bins in each record")
    parser.add_argument("--iterations", type=int, help="override kilo-iterations")
    parser.add_argument("--print-every", type=int, default=1, help="print a grid every Nth set")
    parser.add_argument("--allow-foreign-config", action="store_true",
                        help="capture even though the requested configuration was not applied")
    args = parser.parse_args()

    if args.new_notes:
        if os.path.exists(args.new_notes):
            return "{} already exists; refusing to overwrite field notes".format(args.new_notes)
        shutil.copyfile(capture_log.NOTES_TEMPLATE, args.new_notes)
        print("wrote {}\nFill it in, then pass it with --notes.".format(args.new_notes))
        return 0
    if not args.label:
        parser.error("--label is required")

    period_ms = int(round(args.interval * 1000))
    if not 1 <= period_ms <= 0xFFFF:
        return "--interval must be between 0.001 and 65.535 seconds (period is a 16-bit ms field)"
    if args.print_every < 1:
        return "--print-every must be at least 1"

    client = ZeroMqClient()
    connect(client, args.host)

    try:
        device = client.identify()
    except Exception as exc:
        return "no TMF8829 ZeroMQ server at {}:5557 ({})".format(args.host, exc)

    print("connected to {} serial=0x{:08x} hostType={} fw={}".format(
        args.host, device.deviceSerialNumber, device.hostType, list(device.fwVersion)))

    histograms = 0 if args.no_histograms else 1
    outcome = configure(client, period_ms, histograms, args.iterations)
    cfg, cfg_raw = outcome["actual"], outcome["raw"]
    to_mm = cfg["select"] >= 1
    unit = "mm" if to_mm else "bins"
    print("fp_mode={} period={}ms iterations={} histograms={} nr_peaks={} units={}".format(
        cfg["fp_mode"], cfg["period"], cfg["iterations"], cfg["histograms"], cfg["nr_peaks"], unit))

    foreign = not (client._is_cfg_client and outcome["pre_ok"] and outcome["set_ok"])
    shortfalls = config_shortfalls(cfg, histograms)
    if foreign:
        print("\nThis client does not own the device configuration"
              " (is_cfg_client={} set_pre_config={} set_config={}); the device is running its own"
              " setup.".format(client._is_cfg_client, outcome["pre_ok"], outcome["set_ok"]))
        for key, want, got in config_preferences(outcome["requested"], cfg):
            print("   {:18s} wanted {:<8} running {}".format(key, want, got))

    if shortfalls:
        print("\nThe device is not producing what this capture needs:")
        for key, want, got in shortfalls:
            print("   {:18s} need {:<8} got {}".format(key, want, got))
        for item in describe_lost_channels(cfg):
            print("   - {}".format(item))
        if not args.allow_foreign_config:
            client.leave()
            client.disconnect_local()
            return ("refusing to capture: the data would be missing channels the offline analysis "
                    "needs.\nEither own the configuration (the first client to identify() gets it, "
                    "so close any other client and retry), or configure the device itself -- the "
                    "vendor client on a linux-host EVM reads /cfg_client.json, where "
                    '{"preconfig": "' + PRE_CONFIG_NAME + '"} plus a measure_cfg block will make '
                    "the device come up correctly on its own.\nPass --allow-foreign-config to "
                    "capture anyway; the log records the true configuration either way.")
        print("\n--allow-foreign-config given: capturing with channels missing.")
    elif foreign:
        print("The running configuration provides everything this capture needs; logging it as-is.")

    notes = capture_log.read_notes(args.notes) if args.notes else None
    if notes:
        complaint = capture_log.notes_complaint(notes)
        if complaint:
            print("WARNING: field notes at {} look unfinished ({}). The notes are the only part of "
                  "a capture that cannot be recovered later.".format(args.notes, complaint))
    elif not args.no_log:
        print("no --notes given; nothing will record what the sensor was pointed at beyond "
              "--label. Start one with --new-notes <path>.")

    writer = None
    if not args.no_log:
        os.makedirs(args.log_dir, exist_ok=True)
        path = capture_log.log_path(args.log_dir, args.label, device.deviceSerialNumber)
        writer = capture_log.CaptureWriter(path, decode_histograms=args.decode_histograms)
        writer.write_header(
            label=args.label,
            preconfig=PRE_CONFIG_NAME,
            is_cfg_client=bool(client._is_cfg_client),
            device=capture_log.device_info_to_dict(device),
            config=cfg,
            config_raw=base64.b64encode(bytes(cfg_raw)).decode("ascii"),
            config_applied=not foreign,
            config_complete=not shortfalls,
            config_requested=outcome["requested"],
            config_shortfalls=[
                {"field": k, "needed": want, "actual": got} for k, want, got in shortfalls
            ],
            set_pre_config_ok=outcome["pre_ok"],
            set_config_ok=outcome["set_ok"],
            notes=notes,
        )
        print("logging to {}".format(path))

    if not client.start_measurement():
        if writer:
            writer.close()
        client.leave()
        return "server reports no measurement running (logger-only client)"

    captured = 0
    started = time.monotonic()
    try:
        while args.count == 0 or captured < args.count:
            # The device paces itself at `period`, so this blocks until the next set.
            payload = client.get_result_data(timeout=args.interval + 10.0)
            captured += 1
            if writer:
                decoded = writer.write_set(payload)
            else:
                try:
                    decoded = capture_log.decode_set(payload, args.decode_histograms)
                except Exception as exc:
                    decoded = {"error": "{}: {}".format(type(exc).__name__, exc)}
            if captured % args.print_every == 0:
                print_set(decoded, to_mm, args.peak, unit)
    except KeyboardInterrupt:
        pass
    except TimeoutError as exc:
        print("stopped: {}".format(exc))
    finally:
        client.stop_measurement()
        client.leave()
        client.disconnect_local()
        if writer:
            writer.close()

    elapsed = time.monotonic() - started
    print("captured {} frame sets in {:.1f}s ({:.2f} sets/s)".format(
        captured, elapsed, captured / elapsed if elapsed else 0.0))
    if writer:
        print("log {} ({:.1f} MB raw payload)".format(writer.path, writer.bytes_raw / 1e6))
        if writer.decode_errors:
            print("{} sets failed to decode; their raw bytes are in the log".format(
                writer.decode_errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
