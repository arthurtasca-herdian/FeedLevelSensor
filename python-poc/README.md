# python-poc

Captures measurement data from a real TMF8829 EVM for offline analysis. Runs as a ZeroMQ client against
the vendor's server on the machine the EVM is cabled to.

# System Setup

## PoC configuration

1. under [TMF8829 landing page](https://ams-osram.com/products/sensor-solutions/direct-time-of-flight-sensors-dtof/ams-tmf8829-48x32-multi-zone-time-of-flight-sensor), download and extract:
    * TMF8829_Driver_Python_v2026-02-02.zip
    * TMF8829_Driver_ZMQ_Server_Client_EXE_v2.2.7.zip
    * TMF8829_App_json_logfile_viewer_v1.2.zip
    * TMF8829_EVM_GUI.exe
1. Extract the Python driver into `./driver` — every script here requires it:
   ```
   unzip TMF8829_Driver_Python_v*.zip -d driver
   ```
   The archive root holds `tmf8829/`, `packages/` and `requirements.txt`, giving the layout below.
1. Install dependencies with `uv sync`. Do **not** install `driver/requirements.txt`: it pins
   `corefw_c-*-win_amd64.whl`, which cannot install on Linux. The scripts import only the driver's
   pure-ctypes modules, so `pyzmq` is the only dependency.
1. Run `TMF8829_Driver_ZMQ_Server_Client_EXE/tmf8829_zeromq_server.exe` every time you want to connect to the pre-flashed board
1. Run `TMF8829_Driver_ZMQ_Server_Client_EXE/tmf8829_zeromq_client.exe` for an immediate testing connection

An encapsulated EVM with a linux host runs the server on the device itself; pass its address with `--host 169.254.0.2` instead of starting a local server.

Only the first client to connect may change the device configuration; every later client is read-only until that client disconnects or the server restarts. Start the client that needs to configure first.

## Required structure

```
python-poc/
├── configs/
├── captures/                 created on first capture
└── driver/
    ├── packages/
    ├── requirements.txt
    └── tmf8829/
        ├── aos_com/
        ├── utilities/
        └── zeromq/
```
