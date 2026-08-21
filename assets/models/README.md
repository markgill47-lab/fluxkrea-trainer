# Vendored model weights

Small model files that ship with the repository rather than being fetched
at runtime, so the Olympus install script needs no extra network access
(doc 04, "detection").

## `face_detection_yunet_2023mar.onnx`

The YuNet face detector used by `fk dataset detect` and `fk dataset mask`.
About 350KB. It is **not** shipped with the `opencv-python` pip package,
so it has to be placed here once:

```bash
curl -L -o assets/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

**Licence.** YuNet is distributed by the OpenCV Model Zoo under the MIT
Licence, which permits redistribution with the licence and copyright
notice. It is vendored here on that basis. Source:
<https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>.

Check it landed with `fk node status` — the detector line reads `ready`
once the file is in place, and `unavailable` until then.

Everything else in the masking pipeline works without it: boxes can be
drawn by hand (`fk dataset boxes`), and masks export from stored boxes
either way. Only automatic detection needs the weights.

Nothing large belongs in this folder. Training checkpoints live wherever
`backends.output_root` points, never in the repository.
