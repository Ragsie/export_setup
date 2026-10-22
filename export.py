#!/usr/bin/env python3
"""
NuroMow - Qualcomm QNN Native Offline Compiler Pipeline (v13)
============================================================
This script takes your trained YOLO .pt model, exports it to ONNX, and uses
Qualcomm's QNN SDK on the x86_64 server to compile a Hexagon HTP Context Binary (.bin) offline:

Workflow:
  Step 1: PyTorch (.pt) -> ONNX (.onnx) via Ultralytics
  Step 2: ONNX -> QNN C++ & binary weights (.cpp & .bin) via qnn-onnx-converter
  Step 3: QNN C++ -> x86_64 Model Shared Library (.so) via qnn-model-lib-generator (-t x86_64-linux-clang)
  Step 4: x86_64 Shared Library (.so) -> Hexagon HTP NPU Context Binary (.bin) via qnn-context-binary-generator
  Step 5: Deploy package (.tar.gz) with .bin + QNN runtime libraries from the SAME SDK + test script

IMPORTANT: The Radxa Dragon Q6A (QCS6490) has a Hexagon v68 NPU, which ONLY runs quantized
(INT8/INT16) models. That is why calibration images (--calib_dir) are required in step 2.

IMPORTANT 2: A context binary can only be run by a QNN runtime of the SAME or NEWER version
than the SDK that built it. The deploy package therefore includes the runtime libraries from
the server's SDK, so the version on the SBC does not matter. Run device/qnn_info.py on
the SBC to see which version it has.
"""

import os
import sys
import argparse
import shutil
import subprocess
import json
import glob
import re
import shlex
import tarfile

def run_cmd(cmd, env, step_name):
    """Runs a QNN tool and shows both stdout and stderr on failure (QNN logs mostly to stdout)."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error during {step_name} (exit code {result.returncode})")
        print("----- stdout -----")
        print(result.stdout[-5000:])
        print("----- stderr -----")
        print(result.stderr[-5000:])
        return False
    return True

SDK_MARKER = os.path.join("bin", "x86_64-linux-clang", "qnn-context-binary-generator")

def resolve_sdk_root(path, max_depth=4):
    """
    SDKs are often unpacked as e.g. snpe/qairt/2.42.0.251225/. If 'path' is not itself the
    SDK root, search downwards for the folder with bin/x86_64-linux-clang/qnn-context-binary-generator.
    If several are found, the newest version is chosen (list them all with --sdk_info).
    """
    found = find_sdk_roots(path, max_depth)
    if not found:
        return path
    return sorted(found, key=lambda r: parse_version(get_sdk_version(r)) or (0,))[-1]

def find_sdk_roots(path, max_depth=4):
    if os.path.exists(os.path.join(path, SDK_MARKER)):
        return [path]
    found = []
    base_depth = path.rstrip(os.sep).count(os.sep)
    for root, dirs, _ in os.walk(path):
        if root.count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
        if os.path.exists(os.path.join(root, SDK_MARKER)):
            found.append(root)
            dirs[:] = []
    return found

def find_qnn_root(custom_path=None):
    if custom_path:
        if not os.path.exists(custom_path):
            print(f"❌ QNN SDK not found at --qnn_root: {custom_path}")
            return None
        return resolve_sdk_root(custom_path)
    if os.environ.get("QNN_SDK_ROOT") and os.path.exists(os.environ["QNN_SDK_ROOT"]):
        return resolve_sdk_root(os.environ["QNN_SDK_ROOT"])
    candidates = [
        # QAIRT 2.42 matches the Radxa Dragon Q6A's official runtime (2.42.0.251225)
        "/home/jovyan/work/qairt/2.42.0.251225",
        "/home/jovyan/work/qairt",
        "/home/jovyan/work/qnn",
        "/home/jovyan/work/snpe/qnn",
        "/home/jovyan/work/snpe",
        "/opt/qcom/qnn",
        "/opt/qcom/snpe"
    ]
    for c in candidates:
        if os.path.exists(c):
            return resolve_sdk_root(c)
    print("❌ Could not find the QNN SDK. Use --qnn_root or set QNN_SDK_ROOT.")
    return None

def setup_qnn_env(qnn_sdk_root):
    env = os.environ.copy()
    # Clear Conda x86_64 flags that break C++ compilation
    for var in ["CXXFLAGS", "CFLAGS", "CPPFLAGS", "LDFLAGS"]:
        env.pop(var, None)

    qnn_bin = os.path.join(qnn_sdk_root, "bin/x86_64-linux-clang")
    qnn_lib = os.path.join(qnn_sdk_root, "lib/x86_64-linux-clang")

    if not os.path.exists(qnn_bin):
        qnn_bin = os.path.join(qnn_sdk_root, "bin")
    if not os.path.exists(qnn_lib):
        qnn_lib = os.path.join(qnn_sdk_root, "lib")

    env["QNN_SDK_ROOT"] = qnn_sdk_root
    env["PATH"] = f"{qnn_bin}:{env.get('PATH', '')}"
    ld_path = [qnn_lib] + [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    # QNN's Python modules (libPyIrGraph.so) are linked against libpython3.X.so, which lives in
    # the QNN Python's lib folder. It goes last so it doesn't shadow the system libraries.
    py_libdir = python_libdir(qnn_python(qnn_sdk_root))
    if py_libdir and py_libdir not in ld_path:
        ld_path.append(py_libdir)
    env["LD_LIBRARY_PATH"] = ":".join(ld_path)
    env["PYTHONPATH"] = f"{qnn_sdk_root}/lib/python:{env.get('PYTHONPATH', '')}"
    return env

_libdir_cache = {}

def python_libdir(py):
    """The folder containing libpython3.X.so for a given Python (e.g. /opt/conda/envs/qairt/lib)."""
    if py not in _libdir_cache:
        try:
            r = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')"],
                               capture_output=True, text=True, timeout=30)
            _libdir_cache[py] = r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            _libdir_cache[py] = None
    return _libdir_cache[py]

def safe_model_name(name):
    """QNN uses the file name as the C++ graph name, so only letters, digits and _ are safe."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # A C++ name must not start with a digit
    if not name or name[0].isdigit():
        name = f"m_{name}"
    return name

def find_tool(qnn_sdk_root, name):
    for p in (os.path.join(qnn_sdk_root, "bin/x86_64-linux-clang", name),
              os.path.join(qnn_sdk_root, "bin", name)):
        if os.path.exists(p):
            return p
    print(f"❌ Tool '{name}' was not found in {qnn_sdk_root}.")
    return None

def tool_cmd(tool_path, qnn_sdk_root=None):
    """
    The QNN tools are a mix of Python scripts, shell scripts and binaries.
    Python scripts are run with the Python that matches the SDK (see qnn_python); the rest run directly.
    """
    with open(tool_path, "rb") as f:
        first_line = f.readline(200)
    if first_line.startswith(b"#!") and b"python" in first_line:
        # QNN_PYTHON can point to the Python version your SDK requires (see the SDK docs)
        return [qnn_python(qnn_sdk_root), tool_path]
    return [tool_path]

def sdk_python_versions(qnn_sdk_root):
    """
    Which Python versions the SDK's compiled modules are built for, e.g. {"3.10"}.
    Read from file names like libPyIrGraph.cpython-310-x86_64-linux-gnu.so, or from which
    libpython3.X.so the modules are linked against (QAIRT 2.42 uses untagged names like libPyIrGraph.so).
    """
    tags = set()
    pattern = os.path.join(qnn_sdk_root, "lib", "python", "**", "*.cpython-3*-x86_64-linux-gnu.so")
    for f in glob.glob(pattern, recursive=True):
        m = re.search(r"cpython-3(\d+)-", os.path.basename(f))
        if m:
            tags.add(f"3.{m.group(1)}")
    if not tags:
        for f in glob.glob(os.path.join(qnn_sdk_root, "lib", "python", "**", "libPy*.so"), recursive=True):
            try:
                with open(f, "rb") as fh:
                    tags.update(f"3.{m.decode()}" for m in re.findall(rb"libpython3\.(\d+)\.so", fh.read()))
            except OSError:
                pass
    return tags

def python_version_of(py):
    try:
        r = subprocess.run([py, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None

def qnn_python(qnn_sdk_root=None):
    """
    Python for QNN's Python tools. Order:
      1. QNN_PYTHON (if set)
      2. A Python on the machine whose version matches the SDK's compiled modules
         (the current one, /usr/bin/python3.X, /opt/conda/envs/*/bin/python)
      3. The current Python
    You don't need to activate any environment - export.py calls the right one itself.
    """
    py = os.environ.get("QNN_PYTHON")
    if py:
        return py
    qnn_sdk_root = qnn_sdk_root or os.environ.get("QNN_SDK_ROOT")
    wanted = sdk_python_versions(qnn_sdk_root) if qnn_sdk_root else set()
    candidates = [sys.executable]
    for v in sorted(wanted, reverse=True):
        candidates += [f"/usr/bin/python{v}", f"/usr/local/bin/python{v}"]
    candidates += sorted(glob.glob("/opt/conda/envs/*/bin/python"))
    candidates += sorted(glob.glob(os.path.expanduser("~/.conda/envs/*/bin/python")))
    if not wanted:
        # Unknown requirement: prefer the old 'qairt' environment if it exists
        for c in candidates:
            if "/envs/qairt/" in c:
                return c
        return sys.executable
    for c in dict.fromkeys(candidates):
        if os.path.exists(c) and python_version_of(c) in wanted:
            return c
    return sys.executable

def check_qnn_python(qnn_sdk_root, after_setup=False):
    """Checks that QNN's Python matches the SDK and has the packages, BEFORE the slow export."""
    py = qnn_python(qnn_sdk_root)
    wanted = sdk_python_versions(qnn_sdk_root)
    have = python_version_of(py)
    if wanted and have not in wanted:
        print(f"❌ The SDK is built for Python {', '.join(sorted(wanted))}, but the best one found is "
              f"{py} (Python {have}).")
        print(f"   Install Python {sorted(wanted)[0]} (e.g. 'conda create -y -n qairt python={sorted(wanted)[0]}')")
        print("   or set QNN_PYTHON to a Python of that version.")
        return False
    env = setup_qnn_env(qnn_sdk_root)
    # Also import QNN's compiled module, so a missing libpython etc. is caught here
    code = ("import numpy, onnx, yaml; import qti.aisw.converters.common; "
            "print(numpy.__version__)")
    r = subprocess.run([py, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        missing = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown error"
        print(f"❌ QNN Python ({py}) cannot import the converter's packages: {missing}")
        if not after_setup:
            print("   Install them with one command:  python3 export.py --setup_python")
        return False
    if r.stdout.strip().startswith("2."):
        print(f"⚠️ QNN Python uses numpy {r.stdout.strip()} - QAIRT 2.42 typically requires numpy<2.")
    print(f"✅ QNN Python: {py} (Python {have}, numpy {r.stdout.strip()})")
    return True

# Packages QNN's onnx converter (QAIRT 2.4x) needs. The versions are pinned because
# the converter does not work with numpy 2 and is tested against onnx 1.16.
QNN_PY_PACKAGES = [
    "setuptools<70",          # QNN scripts use pkg_resources (removed in setuptools >= 81)
    "numpy==1.26.4",
    "onnx==1.16.1",
    "onnxruntime==1.18.1",
    "protobuf==3.20.3",
    "pyyaml", "packaging", "pandas", "scipy", "attrs", "decorator", "six",
    "typing-extensions", "absl-py", "mako", "jsonschema", "tabulate",
]

def setup_qnn_python(qnn_sdk_root):
    """
    Installs the packages QNN's converter needs into the Python that qnn_python() picks.
    The SDK's own check-python-dependency is not used: it rejects Ubuntu 24.04 without installing anything.
    """
    py = qnn_python(qnn_sdk_root)
    print(f"🐍 Installing QNN packages into {py} (Python {python_version_of(py)})...")
    cmd = [py, "-m", "pip", "install"] + QNN_PY_PACKAGES
    print(f"Running: {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        print("❌ pip install failed - see the output above.")
        return False
    return check_qnn_python(qnn_sdk_root, after_setup=True)

VERSION_RE = re.compile(rb"v?(2\.\d{1,2}\.\d{1,2}\.\d{6})\d*")

def parse_version(v):
    """'v2.42.0.251225' -> (2, 42, 0, 251225) so versions can be compared."""
    if not v:
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)(?:\.(\d{1,6}))?", str(v))
    return tuple(int(x or 0) for x in m.groups()) if m else None

def get_sdk_version(qnn_sdk_root):
    """Finds the QAIRT/QNN version: sdk.yaml, the folder name, or the string in libQnnHtp.so."""
    # E.g. /opt/qcom/aistack/qairt/2.42.0.251225
    m = VERSION_RE.search(os.path.realpath(qnn_sdk_root).encode())
    if m:
        return m.group(1).decode()
    yaml_path = os.path.join(qnn_sdk_root, "sdk.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "rb") as f:
            data = f.read()
        m = re.search(rb"(?m)^\s*version:\s*['\"]?v?(\d+\.\d+\.\d+\.\d+)", data) or VERSION_RE.search(data)
        if m:
            return m.group(1).decode()
    for meta in glob.glob(os.path.join(qnn_sdk_root, "*.yaml")) + glob.glob(os.path.join(qnn_sdk_root, "*.txt")):
        with open(meta, "rb") as f:
            m = VERSION_RE.search(f.read(500000))
        if m:
            return m.group(1).decode()
    lib = os.path.join(qnn_sdk_root, "lib/x86_64-linux-clang/libQnnHtp.so")
    if os.path.exists(lib):
        # Ask the library itself for its build id (same method as device/qnn_info.py).
        # Runs in a subprocess with the SDK's LD_LIBRARY_PATH, otherwise libQnnHtp.so
        # cannot find its dependencies (e.g. libc++).
        info_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device", "qnn_info.py")
        if os.path.exists(info_script):
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{os.path.dirname(lib)}:{env.get('LD_LIBRARY_PATH', '')}"
            code = ("import sys, json; sys.path.insert(0, sys.argv[1]); "
                    "from qnn_info import lib_version; print(json.dumps(lib_version(sys.argv[2])))")
            try:
                r = subprocess.run([sys.executable, "-c", code, os.path.dirname(info_script), lib],
                                   env=env, capture_output=True, text=True, timeout=60)
                build_id = json.loads(r.stdout.strip().splitlines()[-1]).get("build_id") if r.stdout.strip() else None
                if build_id:
                    m = re.search(r"\d+\.\d+\.\d+\.\d{1,6}", build_id)
                    return m.group(0) if m else build_id
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass
        with open(lib, "rb") as f:
            found = sorted({x.group(1).decode() for x in VERSION_RE.finditer(f.read())},
                           key=parse_version)
        if found:
            return found[-1]
    return None

def print_sdk_info(path):
    """Diagnostics for the SDK folder: which SDKs, versions and files are present."""
    roots = find_sdk_roots(path) or [path]
    if len(roots) > 1:
        print(f"Found {len(roots)} SDKs under {path}:")
    for root in roots:
        print(f"\nQNN SDK: {root}")
        print(f"Version: {get_sdk_version(root) or 'unknown'}")
        checks = [
            ("bin/x86_64-linux-clang/qnn-onnx-converter", "converter (step 2)"),
            ("bin/x86_64-linux-clang/qnn-model-lib-generator", "lib-generator (step 3)"),
            (SDK_MARKER, "context-binary-generator (step 4)"),
            ("lib/x86_64-linux-clang/libQnnHtp.so", "x86 HTP backend (step 4)"),
            ("lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so", "HTP config extension (step 4)"),
            ("lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so", "V68 skel (deploy)"),
        ]
        for rel, what in checks:
            ok = os.path.exists(os.path.join(root, rel))
            print(f"  {'✅' if ok else '❌'} {what:<34} {rel}")
        targets = [t for t in AARCH64_TARGETS if os.path.isdir(os.path.join(root, "lib", t))]
        print(f"  {'✅' if targets else '❌'} aarch64 runtime folders (deploy)   {', '.join(targets) or 'none'}")
        if not os.path.exists(os.path.join(root, "lib/x86_64-linux-clang/libQnnHtp.so")):
            libs = sorted(glob.glob(os.path.join(root, "**", "libQnnHtp.so"), recursive=True))[:10]
            if libs:
                print("  libQnnHtp.so found elsewhere:")
                for l in libs:
                    print(f"     {l}")
        if len(roots) > 1:
            print("  (use --qnn_root to select exactly this SDK)")

def check_device_version(sdk_version, device_version):
    """
    A context binary from a NEWER SDK than the runtime on the SBC cannot be loaded.
    Returns False if the combination does not work.
    """
    sdk_v, dev_v = parse_version(sdk_version), parse_version(device_version)
    if not dev_v:
        return True
    if not sdk_v:
        print(f"⚠️ Could not read the SDK version - cannot compare with the SBC's {device_version}.")
        return True
    if sdk_v[:3] > dev_v[:3]:
        print(f"❌ SDK {sdk_version} is NEWER than runtime {device_version} on the SBC.")
        print("   Use an SDK with the same/lower version, or use the runtime libraries from")
        print("   the deploy package on the SBC (use --device_qnn_version none to allow this).")
        return False
    if sdk_v[:3] < dev_v[:3]:
        print(f"ℹ️ SDK {sdk_version} is older than the SBC's runtime {device_version} - that's OK.")
    else:
        print(f"✅ SDK {sdk_version} matches the SBC's runtime {device_version}.")
    return True

def split_yolo_outputs(onnx_path, num_classes, out_path=None):
    """
    YOLO packs boxes (0-imgsz px), class scores (0-1) and mask coefficients into one tensor
    (output0). When quantized they then share one scale, and small scores vanish. Here the
    final Concat is cut away, so each part becomes its own output with its own scale:
      boxes [1,4,N]  scores [1,num_classes,N]  mask_coefs [1,32,N]  (+ output1 = protos)
    The app on the SBC must recombine them (or use them directly).
    Returns a list of output names, or None if the model does not have that structure.
    """
    import onnx
    from onnx import helper, shape_inference

    model = onnx.load(onnx_path)
    graph = model.graph
    if not graph.output:
        return None
    target = graph.output[0].name
    producers = {o: n for n in graph.node for o in n.output}
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    concat = producers.get(target)
    if concat is None or concat.op_type != "Concat":
        print(f"ℹ️ {target} does not come from a Concat - output is not split.")
        return None
    axis = next((a.i for a in concat.attribute if a.name == "axis"), None)

    # Flatten nested Concats on the same axis (e.g. concat(concat(box, cls), mc))
    def flatten(tensor):
        node = producers.get(tensor)
        if (node is not None and node.op_type == "Concat" and len(consumers.get(tensor, [])) == 1
                and next((a.i for a in node.attribute if a.name == "axis"), None) == axis):
            return [t for i in node.input for t in flatten(i)]
        return [tensor]
    parts = [t for i in concat.input for t in flatten(i)]

    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        dims = [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    rank = len(shapes.get(target, [])) or 3
    ax = axis if axis is None or axis >= 0 else axis + rank
    channels = [(shapes.get(t) or [None] * rank)[ax] if ax is not None else None for t in parts]
    if None in channels:
        # Fallback: run the model once with zero input and read off the parts' shapes
        try:
            import numpy as np
            import onnxruntime as ort
            probe = onnx.load(onnx_path)
            for t in parts:
                probe.graph.output.append(onnx.ValueInfoProto(name=t))
            sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
            feeds = {}
            for i in sess.get_inputs():
                dims = [d if isinstance(d, int) else 1 for d in i.shape]
                feeds[i.name] = np.zeros(dims, dtype=np.float32)
            res = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
            for t in parts:
                shapes[t] = list(res[t].shape)
            ax = ax if ax is not None else 1
            channels = [shapes[t][ax] for t in parts]
        except Exception as e:
            print(f"ℹ️ Could not get channel counts for {target}'s parts ({e}) - output is not split.")
            return None

    names, used = [], set()
    for c in channels:
        name = {4: "boxes", num_classes: "scores", 32: "mask_coefs"}.get(c, "part")
        if name in used or name == "part":
            name = f"{name}_{len(names)}"
        used.add(name)
        names.append(name)

    # New outputs via Identity, so they get clean names in QNN
    for t, name in zip(parts, names):
        graph.node.append(helper.make_node("Identity", [t], [name], name=f"split_{name}"))
    old_outputs = list(graph.output)
    del graph.output[:]
    elem = old_outputs[0].type.tensor_type.elem_type
    for t, name in zip(parts, names):
        graph.output.append(helper.make_tensor_value_info(name, elem, shapes.get(t)))
    graph.output.extend(old_outputs[1:])
    graph.node.remove(concat)
    onnx.checker.check_model(model)
    onnx.save(model, out_path or onnx_path)
    print(f"✂️ Split {target} {channels} into separate outputs: {', '.join(names)}")
    return names + [o.name for o in old_outputs[1:]]

def parse_imgsz(imgsz):
    """
    Input size as (height, width). Accepts 800, "800", "448x800", "448,800" or (448, 800).
    Both sides must be a multiple of 32 (YOLO's largest stride).
    A rectangular input (e.g. 448x800 for a 16:9 camera) spares the NPU from the letterbox padding;
    the model does not need retraining, since it is fully convolutional.
    """
    if isinstance(imgsz, (list, tuple)):
        h, w = (int(imgsz[0]), int(imgsz[-1]))
    else:
        parts = [p for p in re.split(r"[x,X* ]+", str(imgsz).strip()) if p]
        h, w = (int(parts[0]), int(parts[-1]))
    for side in (h, w):
        if side <= 0 or side % 32:
            raise ValueError(f"imgsz {h}x{w}: both sides must be a positive multiple of 32 "
                             f"(e.g. 448x800, not 450x800)")
    return h, w

def export_pt_to_onnx(pt_path, imgsz=800, out_dir="/home/jovyan/work/exported_models", base_name=None,
                      split_outputs=True, max_calib_images=200, per_channel=True, converter_args=None):
    """Returns (onnx path, info dict with class names and output names)."""
    print(f"\n📦 [Step 1/5] Exporting {pt_path} to ONNX...", flush=True)
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
    base_name = safe_model_name(base_name)
    onnx_target = os.path.join(out_dir, f"{base_name}.onnx")

    model = YOLO(pt_path)
    class_names = [model.names[i] for i in sorted(model.names)]
    h, w = parse_imgsz(imgsz)
    onnx_file = model.export(format="onnx", imgsz=[h, w], opset=17, simplify=True)

    if onnx_file != onnx_target and os.path.exists(onnx_file):
        shutil.move(onnx_file, onnx_target)

    info = {"class_names": class_names, "outputs_split": False}
    if split_outputs:
        # Keep the unsplit ONNX for comparison and CPU fallback
        shutil.copy2(onnx_target, os.path.join(out_dir, f"{base_name}_full.onnx"))
        names = split_yolo_outputs(onnx_target, len(class_names))
        if names:
            info.update(outputs_split=True, outputs=names)
    print(f"✅ ONNX model ready: {onnx_target}")
    return onnx_target, info

def letterbox(img, imgsz):
    """Same resize + padding Ultralytics uses during inference. imgsz = 800 or (height, width)."""
    import cv2
    th, tw = parse_imgsz(imgsz)
    h, w = img.shape[:2]
    r = min(th / h, tw / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = (th - new_h) // 2
    left = (tw - new_w) // 2
    return cv2.copyMakeBorder(img, top, th - new_h - top, left, tw - new_w - left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))

def create_calibration_data(calib_dirs, imgsz, out_dir, max_images=200):
    """
    Creates .raw files (float32, NHWC, RGB, 0-1) + input_list.txt for quantization.
    calib_dirs can be one folder or a list (e.g. train/images and valid/images).
    Calibration is not evaluation, so training images are fine - they just need to resemble what
    the robot sees. 100-200 images give the most stable quantization scales.
    """
    import cv2
    import numpy as np

    if isinstance(calib_dirs, str):
        calib_dirs = [calib_dirs]
    print(f"\n🖼️ Creating calibration data from {', '.join(calib_dirs)}...", flush=True)
    images = []
    for d in calib_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images += glob.glob(os.path.join(d, ext))
            images += glob.glob(os.path.join(d, ext.upper()))
    images = sorted(set(images))
    if not images:
        print(f"❌ No images found in {', '.join(calib_dirs)}")
        return None
    if len(images) > max_images:
        # Evenly spread over the whole dataset instead of the first N in alphabetical order
        idx = np.linspace(0, len(images) - 1, max_images).round().astype(int)
        images = [images[i] for i in sorted(set(idx))]
    if len(images) < 100:
        print(f"⚠️ Only {len(images)} calibration images - 100-200 give more accurate quantization.")

    raw_dir = os.path.join(out_dir, "calib_raw")
    shutil.rmtree(raw_dir, ignore_errors=True)  # no stale images from a previous run
    os.makedirs(raw_dir, exist_ok=True)
    raw_paths = []
    for i, path in enumerate(images):
        img = cv2.imread(path)
        if img is None:
            continue
        img = letterbox(img, imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # HWC
        raw_path = os.path.abspath(os.path.join(raw_dir, f"calib_{i:04d}.raw"))
        img.tofile(raw_path)
        raw_paths.append(raw_path)

    input_list = os.path.abspath(os.path.join(out_dir, "input_list.txt"))
    with open(input_list, "w") as f:
        f.write("\n".join(raw_paths) + "\n")
    print(f"✅ {len(raw_paths)} calibration images ready: {input_list}")
    return input_list

def run_qnn_onnx_converter(onnx_path, temp_cpp_dir, qnn_sdk_root, input_list, act_bitwidth=16,
                           per_channel=True, extra_args=None):
    print(f"\n🔄 [Step 2/5] Converting ONNX to QNN C++ source & weights (.cpp / .bin)...", flush=True)
    os.makedirs(temp_cpp_dir, exist_ok=True)

    converter_bin = find_tool(qnn_sdk_root, "qnn-onnx-converter")
    if not converter_bin:
        return None, None

    base_name = os.path.splitext(os.path.basename(onnx_path))[0]
    output_cpp = os.path.join(temp_cpp_dir, f"{base_name}.cpp")
    output_bin = os.path.join(temp_cpp_dir, f"{base_name}.bin")

    env = setup_qnn_env(qnn_sdk_root)
    cmd = tool_cmd(converter_bin, qnn_sdk_root) + [
        "-i", onnx_path,
        "-o", output_cpp,
        # Quantization is required on Hexagon v68 (QCS6490)
        "--input_list", input_list,
        "--act_bitwidth", str(act_bitwidth),
        "--weights_bitwidth", "8",
        "--bias_bitwidth", "32",
    ]
    if per_channel:
        # Separate scale per filter channel instead of per layer - typically helps YOLO on HTP
        cmd.append("--use_per_channel_quantization")
    if extra_args:
        cmd += extra_args

    if not run_cmd(cmd, env, "qnn-onnx-converter"):
        return None, None
    if not (os.path.exists(output_cpp) and os.path.exists(output_bin)):
        print(f"❌ Error: qnn-onnx-converter did not produce {output_cpp} / {output_bin}")
        return None, None
    print(f"✅ Generated QNN C++ code: {output_cpp}")
    print(f"✅ Generated QNN binary weights: {output_bin}")
    return output_cpp, output_bin

def run_qnn_model_lib_generator(cpp_path, bin_path, out_dir, qnn_sdk_root):
    """
    Builds an x86_64 shared library (.so), since qnn-context-binary-generator
    runs on x86_64 and must be able to dlopen() the library into its own memory.
    """
    target_arch = "x86_64-linux-clang"
    print(f"\n⚙️ [Step 3/5] Building x86_64 Model Shared Library (.so) for offline HTP build...", flush=True)

    lib_gen_bin = find_tool(qnn_sdk_root, "qnn-model-lib-generator")
    if not lib_gen_bin:
        return None

    env = setup_qnn_env(qnn_sdk_root)
    cmd = tool_cmd(lib_gen_bin, qnn_sdk_root) + [
        "-c", cpp_path,
        "-b", bin_path,
        "-o", out_dir,
        "-t", target_arch
    ]

    if not run_cmd(cmd, env, "qnn-model-lib-generator"):
        return None

    base_name = os.path.splitext(os.path.basename(cpp_path))[0]
    possible_so_paths = [
        os.path.join(out_dir, target_arch, "lib", f"lib{base_name}.so"),
        os.path.join(out_dir, target_arch, f"lib{base_name}.so"),
        os.path.join(out_dir, "lib", f"lib{base_name}.so"),
        os.path.join(out_dir, f"lib{base_name}.so")
    ]

    found_so = None
    for p in possible_so_paths:
        if os.path.exists(p):
            found_so = p
            break

    if not found_so:
        for root, _, files in os.walk(out_dir):
            if f"lib{base_name}.so" in files:
                found_so = os.path.join(root, f"lib{base_name}.so")
                break

    if found_so:
        print(f"🎉 Compiled x86_64 Shared Library (.so) ready: {found_so}")
        return found_so
    else:
        print("❌ Error: Could not find the generated .so file.")
        return None

def write_htp_config(out_dir, qnn_sdk_root, graph_name, dsp_arch, soc_model, vtcm_mb=2):
    """
    Creates the backend config so the context binary is compiled for the right NPU.
    QCS6490 (Radxa Dragon Q6A): dsp_arch v68, soc_model 35, 2 MB VTCM.
    Without vtcm_mb the generator picks 4 MB, and then the NPU rejects the context binary
    ("Request feature vtcm size with value 4194304 unsupported").
    """
    htp_config = {
        "graphs": [{"graph_names": [graph_name], "O": 3, "vtcm_mb": vtcm_mb}],
        "devices": [{"dsp_arch": dsp_arch, "soc_model": soc_model}],
    }
    htp_config_path = os.path.abspath(os.path.join(out_dir, "htp_config.json"))
    with open(htp_config_path, "w") as f:
        json.dump(htp_config, f, indent=2)

    ext_lib = os.path.join(qnn_sdk_root, "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so")
    if not os.path.exists(ext_lib):
        print(f"❌ Error: {ext_lib} does not exist - cannot set dsp_arch/soc_model.")
        return None
    backend_config = {
        "backend_extensions": {
            "shared_library_path": ext_lib,
            "config_file_path": htp_config_path,
        }
    }
    backend_config_path = os.path.abspath(os.path.join(out_dir, "htp_backend_config.json"))
    with open(backend_config_path, "w") as f:
        json.dump(backend_config, f, indent=2)
    return backend_config_path

def run_qnn_context_binary_generator(x86_64_so_path, out_dir, qnn_sdk_root, dsp_arch="v68", soc_model=35,
                                     vtcm_mb=2):
    """
    Runs the x86_64 qnn-context-binary-generator with the x86_64 .so library
    and the x86_64 libQnnHtp.so backend to create the Hexagon HTP NPU Context Binary (.bin).
    """
    print(f"\n⚡ [Step 4/5] Generating Hexagon HTP NPU Context Binary (.bin)...", flush=True)

    ctx_gen_bin = find_tool(qnn_sdk_root, "qnn-context-binary-generator")
    if not ctx_gen_bin:
        return None

    backend_so = os.path.join(qnn_sdk_root, "lib/x86_64-linux-clang/libQnnHtp.so")
    if not os.path.exists(backend_so):
        print(f"❌ Error: {backend_so} does not exist.")
        return None

    base_name = os.path.splitext(os.path.basename(x86_64_so_path))[0]
    if base_name.startswith("lib"):
        base_name = base_name[3:]
    output_context_bin = os.path.join(out_dir, f"{base_name}_htp_context.bin")
    # Graph name = the name of the .cpp file from qnn-onnx-converter
    config_file = write_htp_config(out_dir, qnn_sdk_root, base_name, dsp_arch, soc_model, vtcm_mb)
    if not config_file:
        return None

    env = setup_qnn_env(qnn_sdk_root)
    # Binary program: must be RUN DIRECTLY, WITHOUT PYTHON
    cmd = [
        ctx_gen_bin,
        "--model", x86_64_so_path,
        "--backend", backend_so,
        "--output_dir", out_dir,
        "--config_file", config_file,
        # Without the .bin extension - the tool adds it itself
        "--binary_file", f"{base_name}_htp_context"
    ]

    if not run_cmd(cmd, env, "qnn-context-binary-generator"):
        return None
    if not os.path.exists(output_context_bin):
        print(f"❌ Error: {output_context_bin} was not created.")
        return None
    print(f"🎉 Hexagon NPU Context Binary (.bin) created successfully: {output_context_bin}")
    return output_context_bin

# aarch64 Linux folders in the SDK, in order of priority (the Radxa runs Ubuntu)
AARCH64_TARGETS = ["aarch64-ubuntu-gcc9.4", "aarch64-oe-linux-gcc11.2",
                   "aarch64-oe-linux-gcc9.3", "aarch64-oe-linux-gcc8.2"]

RUN_SCRIPT = """#!/bin/bash
# NuroMow - test the context binary on the Radxa Dragon Q6A with the runtime from the same SDK.
# Usage: ./run_on_device.sh            (runs qnn-net-run on one calibration image)
#        source ./run_on_device.sh env (only sets the environment variables for your own app)
set -e
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export LD_LIBRARY_PATH="$DIR/lib:$LD_LIBRARY_PATH"
# The skel from the package MUST be found first, so it matches libQnnHtp.so
export ADSP_LIBRARY_PATH="$DIR/dsp;/usr/lib/dsp/cdsp;/usr/lib/dsp;/usr/lib/rfsa/adsp;/dsp"
unset LD_PRELOAD
[ "$1" = "env" ] && return 0 2>/dev/null

[ -f "$DIR/qnn_info.py" ] && python3 "$DIR/qnn_info.py" "$DIR" || true
if [ ! -x "$DIR/bin/qnn-net-run" ]; then
    echo "qnn-net-run is missing from the package - cannot test."; exit 1
fi
cd "$DIR"
rm -rf "$DIR/test_output"   # no stale .raw files from a previous run
echo "$DIR/test_input.raw" > "$DIR/test_input_list.txt"
"$DIR/bin/qnn-net-run" \\
    --backend "$DIR/lib/libQnnHtp.so" \\
    --retrieve_context "$DIR/{model_bin}" \\
    --input_list "$DIR/test_input_list.txt" \\
    --output_dir "$DIR/test_output"
echo "✅ NPU inference worked! Output in $DIR/test_output"
"""

def build_deploy_package(htp_bin, out_dir, qnn_sdk_root, sdk_version, model_info, input_list,
                         dsp_arch="v68", aarch64_target=None):
    """
    Collects everything the SBC needs into one folder + .tar.gz:
      <name>_htp_context.bin, lib/ (libQnnHtp.so, Stub, libQnnSystem.so), dsp/ (V68 Skel),
      bin/qnn-net-run, qnn_info.py, run_on_device.sh, model_info.json, test_input.raw
    The runtime comes from the SAME SDK that built the .bin file, so the versions always match.
    """
    print("\n📦 [Step 5/5] Building deploy package for the Radxa Dragon Q6A...", flush=True)
    arch_num = dsp_arch.lstrip("vV")
    targets = [aarch64_target] if aarch64_target else AARCH64_TARGETS
    target = next((t for t in targets if os.path.isdir(os.path.join(qnn_sdk_root, "lib", t))), None)
    if not target:
        print(f"❌ Found no aarch64 lib folder in {qnn_sdk_root}/lib (tried {targets}).")
        return None
    print(f"   Using runtime from lib/{target}")

    name = os.path.splitext(os.path.basename(htp_bin))[0]
    pkg_dir = os.path.join(out_dir, f"{name}_deploy")
    if os.path.exists(pkg_dir):
        shutil.rmtree(pkg_dir)
    for sub in ("lib", "dsp", "bin"):
        os.makedirs(os.path.join(pkg_dir, sub))

    lib_dir = os.path.join(qnn_sdk_root, "lib", target)
    skel_dir = os.path.join(qnn_sdk_root, "lib", f"hexagon-v{arch_num}", "unsigned")
    files = [
        (os.path.join(lib_dir, "libQnnHtp.so"), "lib", True),
        (os.path.join(lib_dir, f"libQnnHtpV{arch_num}Stub.so"), "lib", True),
        (os.path.join(lib_dir, "libQnnSystem.so"), "lib", True),
        (os.path.join(lib_dir, "libQnnHtpNetRunExtensions.so"), "lib", False),
        (os.path.join(skel_dir, f"libQnnHtpV{arch_num}Skel.so"), "dsp", True),
        (os.path.join(qnn_sdk_root, "bin", target, "qnn-net-run"), "bin", False),
    ]
    for src, sub, required in files:
        if not os.path.exists(src):
            if required:
                print(f"❌ Missing {src} - cannot build the deploy package.")
                return None
            print(f"⚠️ Skipping (does not exist): {src}")
            continue
        shutil.copy2(src, os.path.join(pkg_dir, sub, os.path.basename(src)))

    shutil.copy2(htp_bin, os.path.join(pkg_dir, os.path.basename(htp_bin)))
    here = os.path.dirname(os.path.abspath(__file__))
    info_script = os.path.join(here, "device", "qnn_info.py")
    if os.path.exists(info_script):
        shutil.copy2(info_script, os.path.join(pkg_dir, "qnn_info.py"))
    else:
        print(f"⚠️ {info_script} is missing - the package gets no qnn_info.py. "
              "Copy the whole repo (incl. device/) alongside export.py.")

    # First calibration image as test input (float32 NHWC, the same format qnn-net-run expects)
    with open(input_list) as f:
        first_raw = f.readline().strip()
    if first_raw and os.path.exists(first_raw):
        shutil.copy2(first_raw, os.path.join(pkg_dir, "test_input.raw"))

    run_sh = os.path.join(pkg_dir, "run_on_device.sh")
    with open(run_sh, "w") as f:
        f.write(RUN_SCRIPT.format(model_bin=os.path.basename(htp_bin)))
    os.chmod(run_sh, 0o755)

    meta = dict(model_info, qnn_sdk_version=sdk_version, aarch64_target=target, dsp_arch=dsp_arch,
                context_binary=os.path.basename(htp_bin))
    with open(os.path.join(pkg_dir, "model_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    tar_path = pkg_dir + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pkg_dir, arcname=os.path.basename(pkg_dir))
    print(f"🎉 Deploy package ready: {tar_path}")
    print(f"   On the SBC:  tar xzf {os.path.basename(tar_path)} && "
          f"cd {os.path.basename(pkg_dir)} && ./run_on_device.sh")
    return tar_path

def export_pipeline(pt_path, calib_dir, out_dir="/home/jovyan/work/exported_models", imgsz=800,
                    qnn_root=None, act_bitwidth=16, dsp_arch="v68", soc_model=35, base_name=None,
                    device_qnn_version=None, package=True, aarch64_target=None, vtcm_mb=2,
                    split_outputs=True, max_calib_images=200, per_channel=True, converter_args=None):
    """
    The whole chain .pt -> ONNX -> QNN C++ -> x86_64 .so -> Hexagon context binary (.bin) -> deploy package.
    Used both from the command line here and from mlops_master.py after training.
    Returns the path to the deploy package (.tar.gz), or the .bin if package=False, or None on failure.
    """
    try:
        imgsz = parse_imgsz(imgsz)
    except ValueError as e:
        print(f"❌ {e}")
        return None
    print(f"📐 Model input: {imgsz[0]}x{imgsz[1]} (height x width)")

    if not os.path.isfile(pt_path):
        print(f"❌ Model not found: {os.path.abspath(pt_path)}")
        candidates = sorted(glob.glob("/home/jovyan/work/runs/**/weights/best.pt", recursive=True)
                            + glob.glob("**/best.pt", recursive=True),
                            key=os.path.getmtime, reverse=True)
        if candidates:
            print("   Models found (newest first):")
            for c in dict.fromkeys(os.path.abspath(c) for c in candidates[:5]):
                print(f"     {c}")
        return None

    qnn_sdk_root = find_qnn_root(qnn_root)
    if not qnn_sdk_root:
        return None
    sdk_version = get_sdk_version(qnn_sdk_root)
    print(f"🎯 Using QNN SDK at: '{qnn_sdk_root}' (version: {sdk_version or 'unknown'})")
    if not check_device_version(sdk_version, device_qnn_version):
        return None
    if not check_qnn_python(qnn_sdk_root):
        return None

    # Check calibration images BEFORE the slow ONNX export
    input_list = create_calibration_data(calib_dir, imgsz, out_dir, max_calib_images)
    if not input_list:
        return None

    # 1. PT -> ONNX
    onnx_path, onnx_info = export_pt_to_onnx(pt_path, imgsz, out_dir, base_name, split_outputs)

    # 2. ONNX -> QNN C++ & binary weights (.cpp & .bin)
    temp_cpp_dir = os.path.join(out_dir, "qnn_generated_cpp")
    cpp_path, bin_path = run_qnn_onnx_converter(onnx_path, temp_cpp_dir, qnn_sdk_root,
                                                input_list, act_bitwidth, per_channel, converter_args)
    if not (cpp_path and bin_path):
        return None

    # 3. QNN C++ -> x86_64 Shared Library (.so)
    x86_so_path = run_qnn_model_lib_generator(cpp_path, bin_path, out_dir, qnn_sdk_root)
    if not x86_so_path:
        return None

    # 4. x86_64 .so -> Hexagon NPU Context Binary (.bin)
    htp_bin = run_qnn_context_binary_generator(x86_so_path, out_dir, qnn_sdk_root,
                                               dsp_arch, soc_model, vtcm_mb)
    if not htp_bin:
        return None
    if not package:
        print("\n🎉 DONE! Your Hexagon NPU Context Binary (.bin) is ready for the Radxa Dragon Q6A!")
        return htp_bin

    # 5. Deploy package with runtime from the same SDK
    model_info = {
        "source_model": os.path.abspath(pt_path),
        # imgsz is a number for square input, otherwise [height, width]
        "imgsz": imgsz[0] if imgsz[0] == imgsz[1] else list(imgsz),
        "input_height": imgsz[0],
        "input_width": imgsz[1],
        "input_shape": [1, imgsz[0], imgsz[1], 3],
        "input_layout": "NHWC",
        "input_dtype": "float32",
        "input_format": "RGB, 0-1, letterbox with (114,114,114) padding",
        "act_bitwidth": act_bitwidth,
        "weights_bitwidth": 8,
        "soc_model": soc_model,
        "vtcm_mb": vtcm_mb,
        "per_channel": per_channel,
        "converter_args": converter_args or [],
        **onnx_info,
    }
    tar_path = build_deploy_package(htp_bin, out_dir, qnn_sdk_root, sdk_version, model_info,
                                    input_list, dsp_arch, aarch64_target)
    if tar_path:
        print("\n🎉 DONE! Copy the deploy package to your Radxa Dragon Q6A.")
    return tar_path

def main():
    parser = argparse.ArgumentParser(description="NuroMow - YOLO to QNN Offline Context Binary (.bin) Generator")
    parser.add_argument("--model", type=str, help="Path to your trained YOLO .pt model")
    parser.add_argument("--imgsz", type=str, default="800",
                        help="Input size: 800 (square) or HEIGHTxWIDTH, e.g. 448x800 for a "
                             "16:9 camera (saves ~44%% NPU work, no retraining). Multiple of 32.")
    parser.add_argument("--qnn_root", type=str, default=None, help="Path to the Qualcomm QNN SDK")
    parser.add_argument("--out_dir", type=str, default="/home/jovyan/work/exported_models", help="Output dir")
    parser.add_argument("--calib_dir", type=str, nargs="+",
                        help="One or more folders with calibration images, e.g. "
                             "dataset/train/images dataset/valid/images (training images are fine)")
    parser.add_argument("--max_calib_images", type=int, default=200,
                        help="At most this many calibration images (evenly spread), default 200")
    parser.add_argument("--act_bitwidth", type=int, default=16, choices=[8, 16],
                        help="16 = A16W8, recommended for YOLO (box coordinates and scores share an output); "
                             "8 = fastest but often poor accuracy")
    parser.add_argument("--dsp_arch", type=str, default="v68", help="Hexagon architecture (QCS6490 = v68)")
    parser.add_argument("--soc_model", type=int, default=35, help="QNN SoC id (QCS6490 = 35)")
    parser.add_argument("--name", type=str, default=None, help="Output model name (default: the .pt file name)")
    parser.add_argument("--device_qnn_version", type=str, default="2.42.0.251225",
                        help="QNN runtime version on the SBC (from device/qnn_info.py). The export stops if "
                             "the SDK is newer. Default: Radxa's official 2.42.0.251225. "
                             "Use 'none' to allow a newer SDK (requires the deploy package's runtime).")
    parser.add_argument("--vtcm_mb", type=int, default=2,
                        help="VTCM (NPU memory) in MB. QCS6490 has 2 MB; try 1 if 2 is rejected")
    parser.add_argument("--no_per_channel", action="store_true",
                        help="Disable per-channel quantization of weights (on by default: "
                             "measured cosine 0.978 -> 0.998 on class scores)")
    parser.add_argument("--converter_args", type=str, default="",
                        help="Extra arguments for qnn-onnx-converter as a single string, "
                             "e.g. \"--act_quantizer_calibration percentile --percentile_calibration_value 99.99\"")
    parser.add_argument("--no_split_outputs", action="store_true",
                        help="Keep YOLO's combined output0 (boxes+scores+masks in one tensor). "
                             "Default is to split it, so quantization doesn't ruin the scores")
    parser.add_argument("--no_package", action="store_true", help="Only build the .bin, no deploy package")
    parser.add_argument("--aarch64_target", type=str, default=None,
                        help=f"SDK lib folder for the SBC (default: first of {AARCH64_TARGETS})")
    parser.add_argument("--sdk_info", action="store_true", help="Only show SDK path and version, then exit")
    parser.add_argument("--setup_python", action="store_true",
                        help="Install the Python packages QNN's converter needs, then exit")

    args = parser.parse_args()
    if args.setup_python:
        root = find_qnn_root(args.qnn_root)
        sys.exit(0 if root and setup_qnn_python(root) else 1)
    if not args.sdk_info and not (args.model and args.calib_dir):
        parser.error("--model and --calib_dir are required")
    if args.sdk_info:
        root = args.qnn_root or os.environ.get("QNN_SDK_ROOT")
        if not root:
            # find_qnn_root picks automatically; then show everything under the candidate folder
            chosen = find_qnn_root(None)
            if not chosen:
                sys.exit(1)
            root = chosen
        print_sdk_info(root)
        chosen = find_qnn_root(root)
        print(f"\n👉 export.py uses: {chosen} (version {get_sdk_version(chosen) or 'unknown'})")
        wanted = sdk_python_versions(chosen)
        py = qnn_python(chosen)
        print(f"🐍 The SDK's Python modules are for Python {', '.join(sorted(wanted)) or 'unknown'}; "
              f"using {py} (Python {python_version_of(py)})")
        sys.exit(0)
    if args.device_qnn_version and args.device_qnn_version.lower() == "none":
        args.device_qnn_version = None
    result = export_pipeline(args.model, args.calib_dir, args.out_dir, args.imgsz, args.qnn_root,
                             args.act_bitwidth, args.dsp_arch, args.soc_model, args.name,
                             args.device_qnn_version, not args.no_package, args.aarch64_target,
                             args.vtcm_mb, not args.no_split_outputs, args.max_calib_images,
                             not args.no_per_channel, shlex.split(args.converter_args))
    if not result:
        sys.exit(1)

if __name__ == "__main__":
    main()
