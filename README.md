# export_setup

Exports a trained YOLO-seg model (`.pt`) to a Hexagon v68 HTP context binary (`.bin`) for the
Radxa Dragon Q6A (QCS6490). The export runs on an x86_64 Ubuntu/Jupyter server with the
Qualcomm QAIRT/QNN SDK.

```
.pt -> ONNX -> QNN .cpp/.bin (A16W8, calibrated) -> x86_64 .so -> HTP context .bin -> deploy .tar.gz
```

## 1. Find the QNN version on your SBC

A context binary can only be run by a QNN runtime with the **same or a newer** version than the
SDK that built it (otherwise: `Using newer context binary on old SDK`). Run on the Radxa:

```bash
python3 device/qnn_info.py
```

The script shows the version of every `libQnnHtp.so` and `libQnnHtpV68Skel.so` it finds, and
whether `/dev/fastrpc-cdsp` and `libcdsprpc.so` are in place. On the server you can see the SDK
version with:

```bash
python3 export.py --sdk_info
```

## 2. Install QAIRT 2.42 on the server

The Radxa Dragon Q6A officially runs QNN runtime **2.42.0.251225**, so the server uses the same
version. `export.py` looks in `/home/jovyan/work/qairt/2.42.0.251225` automatically and stops if
the SDK is newer than 2.42 (`--device_qnn_version`, default `2.42.0.251225`; `none` disables the
check).

```bash
cd /home/jovyan/work
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip
unzip v2.42.0.251225.zip            # gives qairt/2.42.0.251225/
python3 export.py --sdk_info       # shows which Python the SDK needs and which one is used
python3 export.py --setup_python   # installs the converter's packages into that Python (once)
```

## 3. Export on the server

```bash
python3 export.py --model runs/train/worx_segmentation_v2/weights/best.pt \
  --calib_dir /home/jovyan/work/dataset/train/images /home/jovyan/work/dataset/valid/images \
  --name MowerAIn_seg --imgsz 448x800
```

- `--imgsz 448x800` (height x width): rectangular input for a 16:9 camera. At 800x800, 44 % of
  the input is grey letterbox padding; 448x800 saves the NPU that work without retraining (YOLO is
  fully convolutional, and at the same width objects keep the pixel size they had in training).
  Both sides must be a multiple of 32. `mlops_master.py` uses `EXPORT_IMGSZ = (448, 800)`.
  The app must letterbox to the same size; it is in `model_info.json` (`input_height`/`input_width`).
  Measured on YOLO26n-seg with a 1280x720 image: 14.6 -> 8.1 GFLOPs, same detections as
  Ultralytics' own pipeline (which letterboxes rectangularly itself), half the CPU time.
- Outputs: `boxes` are cx, cy, w, h in input pixels (letterbox coordinates); `scores` are
  sigmoid; no NMS in the model.
- `--calib_dir` takes one or more directories. Calibration is not evaluation, so training images
  are fine; up to `--max_calib_images` (200) are picked, spread evenly.
- YOLO's `output0` (boxes + scores + mask coefficients in one tensor) is split into `boxes`,
  `scores` and `mask_coefs`, so each gets its own quantization scale. Otherwise the scores (0-1)
  drown in the box coordinates (0-800). `output1` (prototype masks) is unchanged. The unsplit model
  is saved as `<name>_full.onnx`. `--no_split_outputs` turns this off. The output names are in
  `model_info.json`.

- `--act_bitwidth 16` (default) = A16W8. 8-bit activations usually give worse accuracy.
- Per-channel weight quantization is on (measured on the NPU: class scores cosine 0.978 -> 0.998,
  38/38 detections identical). `--no_per_channel` turns it off.
- `--device_qnn_version` (default `2.42.0.251225`): stops the export if the SDK is newer than the
  SBC's runtime. `none` allows a newer SDK, but then the app must use the deploy package's runtime.
- `--no_package`: only build the `.bin`.
- You don't need to activate any environment. `export.py` finds a Python that matches the SDK's
  compiled modules by itself (`/usr/bin/python3.X` or `/opt/conda/envs/*/bin/python`).
  `QNN_PYTHON=/path/to/python` forces a specific choice.

`mlops_master.py` calls the same pipeline after every training run.

## 4. The deploy package

`exported_models/<name>_htp_context_deploy.tar.gz` contains:

| File | What |
|---|---|
| `<name>_htp_context.bin` | The model |
| `lib/` | `libQnnHtp.so`, `libQnnHtpV68Stub.so`, `libQnnSystem.so` from **the same SDK** |
| `dsp/libQnnHtpV68Skel.so` | The NPU side, same version |
| `bin/qnn-net-run` | Test tool |
| `run_on_device.sh` | Sets `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` and runs a test |
| `model_info.json` | SDK version, input format (NHWC float32 RGB 0-1, letterbox) |

Because the runtime comes from the same SDK, the SBC's own version doesn't matter.

```bash
scp exported_models/MowerAIn_seg_htp_context_deploy.tar.gz radxa@<ip>:~
# on the Radxa:
tar xzf MowerAIn_seg_htp_context_deploy.tar.gz && cd MowerAIn_seg_htp_context_deploy
./run_on_device.sh            # test on the NPU
source ./run_on_device.sh env # only set the environment variables for your own app
```

## Troubleshooting on the SBC

- `fastrpc_test -a v68` (from `sudo apt install fastrpc libcdsprpc1 fastrpc-test`) must work first.
- `0x80000600` / `Failed to create device: 14001`: skel and runtime have different versions, or
  the DSP side is missing `libc++` symlinks in `/usr/lib/dsp`.
- Only `libcdsprpc.so.1` exists: `sudo ln -s libcdsprpc.so.1 /usr/lib/aarch64-linux-gnu/libcdsprpc.so`.
- `LD_PRELOAD` (e.g. mimalloc) breaks fastrpc; `run_on_device.sh` removes it.

## 5. Check the accuracy (NPU vs. ONNX)

After `./run_on_device.sh` on the Radxa: copy `test_output/` back and compare with ONNX on the server.

```bash
rm -rf /home/jovyan/work/exported_models/test_output   # otherwise scp puts the new one inside the old
scp -r radxa@omnimow:~/modeltest/test_output /home/jovyan/work/exported_models/
python3 compare_outputs.py \
  --onnx /home/jovyan/work/exported_models/MowerAIn_seg_full.onnx \
  --input /home/jovyan/work/exported_models/MowerAIn_seg_htp_context_deploy/test_input.raw \
  --npu_dir /home/jovyan/work/exported_models/test_output
```

Cosine > 0.99 for boxes, class scores, mask coefficients and prototype masks = the quantization is fine.
