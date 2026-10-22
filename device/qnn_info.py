#!/usr/bin/env python3
"""
NuroMow - QNN/QAIRT version check (run on the Radxa Dragon Q6A OR on the server)
============================================================================
A Hexagon context binary (.bin) can ONLY be run by a QNN runtime with the SAME or
NEWER version than the SDK that created it. Otherwise you get errors like
"Using newer context binary on old SDK".

This script finds every libQnnHtp.so on the machine and prints its version,
and also checks the things the HTP NPU requires (fastrpc, skel, SoC).

Usage:
  python3 qnn_info.py                 # search default locations
  python3 qnn_info.py /path/to/dir    # also search here (e.g. your deploy package)
  python3 qnn_info.py --json          # machine-readable output

Requires only the Python 3 standard library.
"""

import ctypes
import glob
import json
import os
import platform
import re
import sys

# QNN build id looks like e.g. "v2.42.0.251225135753_193295" (major.minor.patch.date+time_build)
VERSION_RE = re.compile(rb"v?(2\.\d{1,2}\.\d{1,2}\.\d{6})\d*")

SEARCH_DIRS = [
    "/usr/lib", "/usr/lib/aarch64-linux-gnu", "/usr/local/lib", "/usr/lib/dsp",
    "/opt", "/home", "/root",
]
if os.environ.get("QNN_SDK_ROOT"):
    SEARCH_DIRS.insert(0, os.environ["QNN_SDK_ROOT"])


# --- ctypes structures from QnnInterface.h (only the first fields we need) ---
class QnnVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32), ("patch", ctypes.c_uint32)]


class QnnApiVersion(ctypes.Structure):
    _fields_ = [("core", QnnVersion), ("backend", QnnVersion)]


class QnnInterfaceHead(ctypes.Structure):
    _fields_ = [
        ("backendId", ctypes.c_uint32),
        ("providerName", ctypes.c_char_p),
        ("apiVersion", QnnApiVersion),
        # Start of the function table (QNN_INTERFACE_VER_TYPE)
        ("propertyHasCapability", ctypes.c_void_p),
        ("backendCreate", ctypes.c_void_p),
        ("backendSetConfig", ctypes.c_void_p),
        ("backendGetApiVersion", ctypes.c_void_p),
        ("backendGetBuildId", ctypes.c_void_p),
    ]


def version_via_ctypes(lib_path):
    """Asks the library itself for its build id (the same number qnn-net-run logs)."""
    lib = ctypes.CDLL(lib_path)
    get_providers = lib.QnnInterface_getProviders
    get_providers.restype = ctypes.c_uint64
    providers = ctypes.POINTER(ctypes.POINTER(QnnInterfaceHead))()
    num = ctypes.c_uint32(0)
    if get_providers(ctypes.byref(providers), ctypes.byref(num)) != 0 or num.value == 0:
        return None, None
    head = providers[0].contents
    api = head.apiVersion.backend
    api_str = f"{api.major}.{api.minor}.{api.patch}"
    build_id = None
    if head.backendGetBuildId:
        fn = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p))(head.backendGetBuildId)
        out = ctypes.c_char_p()
        if fn(ctypes.byref(out)) == 0 and out.value:
            build_id = out.value.decode(errors="replace")
    return build_id, api_str


def version_via_strings(lib_path):
    """Fallback: find the version string directly in the file (also works for libs for another CPU)."""
    with open(lib_path, "rb") as f:
        data = f.read()
    found = sorted({m.group(1).decode() for m in VERSION_RE.finditer(data)})
    return found[-1] if found else None


def lib_version(lib_path):
    info = {"path": lib_path, "build_id": None, "backend_api": None, "method": None}
    try:
        build_id, api = version_via_ctypes(lib_path)
        if build_id or api:
            info.update(build_id=build_id, backend_api=api, method="ctypes")
    except (OSError, AttributeError):
        pass  # e.g. an x86_64 lib on ARM or missing dependencies
    if not info["build_id"]:
        try:
            v = version_via_strings(lib_path)
            if v:
                info.update(build_id=v, method=info["method"] or "strings")
        except OSError:
            pass
    return info


def find_libs(name, extra_dirs):
    hits = []
    for d in list(extra_dirs) + SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            # Avoid descending endlessly into large directories
            if root.count(os.sep) - d.count(os.sep) > 7:
                dirs[:] = []
                continue
            if name in files:
                p = os.path.realpath(os.path.join(root, name))
                if p not in hits:
                    hits.append(p)
    return hits


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def normalize(v):
    """'v2.42.0.251225...' -> (2, 42, 0, 251225) so versions can be compared."""
    if not v:
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)(?:\.(\d{1,6}))?", v)
    return tuple(int(x or 0) for x in m.groups()) if m else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv

    report = {
        "machine": platform.machine(),
        "os": read("/etc/os-release") and dict(
            line.split("=", 1) for line in read("/etc/os-release").splitlines() if "=" in line
        ).get("PRETTY_NAME", "").strip('"'),
        "glibc": platform.libc_ver()[1],
        "soc_id": read("/sys/devices/soc0/soc_id"),       # QCS6490 = 497 / 475
        "soc_machine": read("/sys/devices/soc0/machine"),
        "fastrpc_cdsp": os.path.exists("/dev/fastrpc-cdsp"),
        "libcdsprpc": [p for p in glob.glob("/usr/lib/aarch64-linux-gnu/libcdsprpc.so*")],
        "htp_libs": [lib_version(p) for p in find_libs("libQnnHtp.so", args)],
        "v68_skels": [lib_version(p) for p in find_libs("libQnnHtpV68Skel.so", args)],
    }

    if as_json:
        print(json.dumps(report, indent=2))
        return

    print("=== NuroMow QNN info ===")
    print(f"CPU arch       : {report['machine']}")
    print(f"OS             : {report['os']}")
    print(f"glibc          : {report['glibc']}")
    print(f"SoC            : id={report['soc_id']} machine={report['soc_machine']}")
    if report["machine"] == "aarch64":
        print(f"/dev/fastrpc-cdsp : {'✅ present' if report['fastrpc_cdsp'] else '❌ missing (NPU not available)'}")
        print(f"libcdsprpc        : {', '.join(report['libcdsprpc']) or '❌ missing (sudo apt install fastrpc libcdsprpc1)'}")
        if report["libcdsprpc"] and not any(p.endswith("libcdsprpc.so") for p in report["libcdsprpc"]):
            print("   ⚠️ Only versioned libcdsprpc.so.1 found. QNN looks for libcdsprpc.so:")
            print("      sudo ln -s libcdsprpc.so.1 /usr/lib/aarch64-linux-gnu/libcdsprpc.so")

    print("\nlibQnnHtp.so (runtime on the CPU side):")
    if not report["htp_libs"]:
        print("   (none found)")
    for lib in report["htp_libs"]:
        print(f"   {lib['build_id'] or '?':<22} api={lib['backend_api'] or '?':<8} {lib['path']}")

    print("\nlibQnnHtpV68Skel.so (runs on the NPU itself):")
    if not report["v68_skels"]:
        print("   (none found)")
    for lib in report["v68_skels"]:
        print(f"   {lib['build_id'] or '?':<22} {lib['path']}")

    versions = sorted({normalize(l["build_id"]) for l in report["htp_libs"] if normalize(l["build_id"])})
    if versions:
        oldest = ".".join(str(x) for x in versions[0])
        print(f"\n👉 Lowest QNN runtime version found: {oldest}")
        print("   Context binaries must be built with a QAIRT/QNN SDK of the SAME or LOWER version,")
        print("   OR you must use the runtime libraries from the deploy package that export.py creates")
        print("   (then the version always matches).")
        if len(versions) > 1:
            print("   ⚠️ There are several different versions - make sure your app uses the right one")
            print("      (LD_LIBRARY_PATH / ADSP_LIBRARY_PATH).")
    print("\nTip: 'fastrpc_test -a v68' (package fastrpc-test) checks that the NPU responds.")


if __name__ == "__main__":
    main()
