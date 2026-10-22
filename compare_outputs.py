#!/usr/bin/env python3
"""
NuroMow - Compare NPU output (Radxa) with ONNX output (server) on the same image
================================================================================
Checks that the INT8/INT16 quantization has not broken the model.

1. Run ./run_on_device.sh on the Radxa (creates test_output/Result_0/*.raw)
2. Copy the test_output folder back to the server:
     rm -rf /home/jovyan/work/exported_models/test_output
     scp -r radxa@omnimow:~/modeltest/test_output /home/jovyan/work/exported_models/
3. Run on the server:
     python3 compare_outputs.py \\
        --onnx /home/jovyan/work/exported_models/MowerAIn_seg_full.onnx \\
        --input /home/jovyan/work/exported_models/MowerAIn_seg_htp_context_deploy/test_input.raw \\
        --npu_dir /home/jovyan/work/exported_models/test_output

qnn-net-run writes float32 .raw files (dequantized). 4D outputs may be NHWC instead
of ONNX's NCHW; the script tries both and uses whichever matches.
"""

import argparse
import glob
import os
import sys

import numpy as np


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else float("nan")


def load_npu_output(npu_dir, name, ref):
    """Finds <name>.raw (QNN may rename '.'/':' to '_') and reshapes it like the ONNX output."""
    candidates = [name, name.replace(".", "_").replace(":", "_").replace("/", "_"), "_" + name]
    path = None
    for c in candidates:
        hits = glob.glob(os.path.join(npu_dir, "**", f"{c}.raw"), recursive=True)
        if hits:
            path = hits[0]
            break
    if not path:
        return None, None
    data = np.fromfile(path, dtype=np.float32)
    if data.size != ref.size:
        print(f"⚠️ {name}: {data.size} values in {path}, ONNX has {ref.size} - wrong dtype or output?")
        return None, path
    options = {"same layout": data.reshape(ref.shape)}
    if ref.ndim == 4:
        n, c, h, w = ref.shape
        options["NHWC->NCHW"] = data.reshape(n, h, w, c).transpose(0, 3, 1, 2)
    if ref.ndim == 3:
        n, c, k = ref.shape
        options["transposed"] = data.reshape(n, k, c).transpose(0, 2, 1)
    best = max(options.items(), key=lambda kv: cosine(kv[1], ref))
    if best[0] != "same layout":
        print(f"   ({name}: NPU output is {best[0]})")
    return best[1], path


SPLIT_PARTS = ("boxes", "scores", "mask_coefs")


def newest_result_dir(npu_dir):
    """
    qnn-net-run writes to <output_dir>/Result_0/. If there are several (e.g. old runs, or
    'scp -r' having put test_output inside an existing test_output), the newest is used.
    """
    dirs = [d for d in glob.glob(os.path.join(npu_dir, "**", "Result_*"), recursive=True) if os.path.isdir(d)]
    if not dirs:
        return npu_dir
    dirs.sort(key=lambda d: max((os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)),
                                default=os.path.getmtime(d)))
    newest = dirs[-1]
    if len(dirs) > 1:
        print(f"⚠️ Found {len(dirs)} Result folders - using the newest: {newest}")
        print("   Delete the old test_output folder before copying a new one over, to avoid confusion.\n")
    raws = glob.glob(os.path.join(newest, "*.raw"))
    if raws:
        newest_t = max(os.path.getmtime(r) for r in raws)
        stale = [os.path.basename(r) for r in raws if newest_t - os.path.getmtime(r) > 60]
        if stale:
            print(f"⚠️ Old files from a previous run in {newest}: {', '.join(stale)} (ignored if possible)\n")
    return newest


def load_split_npu_output(npu_dir, ref, num_classes):
    """NPU model with split outputs (export.py default): combine boxes+scores+mask_coefs."""
    slices = {"boxes": ref[:, :4], "scores": ref[:, 4:4 + num_classes], "mask_coefs": ref[:, 4 + num_classes:]}
    got, paths = [], []
    for name, part_ref in slices.items():
        if part_ref.shape[1] == 0:
            continue
        part, path = load_npu_output(npu_dir, name, part_ref)
        if part is None:
            return None, None
        got.append(part)
        paths.append(os.path.basename(path))
    print("   (output0 combined from the NPU's split outputs)")
    return np.concatenate(got, axis=1), " + ".join(paths)


def report(label, npu, ref):
    err = np.abs(npu - ref)
    cos = cosine(npu, ref)
    ok = "✅" if cos > 0.99 else ("⚠️" if cos > 0.95 else "❌")
    print(f"  {ok} {label:<26} cosine={cos:.4f}  max_err={err.max():.4f}  mean_err={err.mean():.5f}  "
          f"(ONNX range {ref.min():.3f}..{ref.max():.3f})")
    return cos


def compare_detections(npu, ref, num_classes, conf=0.25):
    """YOLO-seg output0: [4 box, num_classes scores, mask coefficients] x number of proposals."""
    ref_s, npu_s = ref[0, 4:4 + num_classes], npu[0, 4:4 + num_classes]
    ref_keep = ref_s.max(0) > conf
    npu_keep = npu_s.max(0) > conf
    both = ref_keep & npu_keep
    print(f"\n  Proposals above conf {conf}: ONNX={ref_keep.sum()}  NPU={npu_keep.sum()}  both={both.sum()}")
    if both.any():
        same_cls = (ref_s[:, both].argmax(0) == npu_s[:, both].argmax(0)).mean()
        box_err = np.abs(ref[0, :4, both] - npu[0, :4, both]).mean()
        print(f"  Same class on shared proposals: {same_cls * 100:.1f}%   avg. box error: {box_err:.2f} px")
    elif ref_keep.any():
        print("  ❌ NPU finds none of the objects ONNX finds.")
    else:
        print("  ℹ️ ONNX finds no objects in this image - choose a test image with objects.")


def main():
    ap = argparse.ArgumentParser(description="Compare NPU output with ONNX output")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True, help="test_input.raw (float32 NHWC, same as the NPU got)")
    ap.add_argument("--npu_dir", required=True, help="test_output folder from the Radxa")
    ap.add_argument("--num_classes", type=int, default=7)
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    _, c, h, w = inp.shape
    x = np.fromfile(args.input, dtype=np.float32)
    if x.size != c * h * w:
        sys.exit(f"❌ {args.input} has {x.size} values, the model expects {c}x{h}x{w}")
    x = x.reshape(1, h, w, c).transpose(0, 3, 1, 2).copy()  # NHWC -> NCHW
    outs = sess.run(None, {inp.name: x})

    refs = dict(zip([o.name for o in sess.get_outputs()], outs))
    # Split ONNX (boxes/scores/mask_coefs): combine into a single output0 as in the original YOLO
    if "boxes" in refs and "scores" in refs:
        parts = [refs.pop(n) for n in SPLIT_PARTS if n in refs]
        refs = {"output0": np.concatenate(parts, axis=1), **refs}

    result_dir = newest_result_dir(args.npu_dir)
    print(f"Comparing {result_dir} with {os.path.basename(args.onnx)}:\n")
    worst = 1.0
    for name, ref in refs.items():
        npu = path = None
        if name == "output0":
            # New (split) outputs take precedence over any old output0.raw from a previous run
            npu, path = load_split_npu_output(result_dir, ref, args.num_classes)
        if npu is None:
            npu, path = load_npu_output(result_dir, name, ref)
        if npu is None:
            print(f"  ❌ {name}: no usable .raw file found in {args.npu_dir}")
            worst = 0.0
            continue
        print(f"{name} {list(ref.shape)}  <-  {path}")
        if ref.ndim == 3 and ref.shape[1] >= 4 + args.num_classes:
            nc = args.num_classes
            worst = min(worst, report("boxes", npu[:, :4], ref[:, :4]))
            worst = min(worst, report("class scores", npu[:, 4:4 + nc], ref[:, 4:4 + nc]))
            if ref.shape[1] > 4 + nc:
                worst = min(worst, report("mask coefficients", npu[:, 4 + nc:], ref[:, 4 + nc:]))
            compare_detections(npu, ref, nc)
        else:
            worst = min(worst, report("prototype masks" if ref.ndim == 4 else name, npu, ref))
        print()

    if worst > 0.99:
        print("🎉 NPU output matches ONNX - the quantization is fine.")
    elif worst > 0.95:
        print("⚠️ Minor deviations. Test on more images; consider more/better calibration images.")
    else:
        print("❌ Large deviations. Try more calibration images (100-200 from the robot) "
              "or check that the input format (RGB, 0-1, letterbox) is the same.")


if __name__ == "__main__":
    main()
