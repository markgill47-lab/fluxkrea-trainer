# 04 — face masking

The feature that started the rewrite.

## Problem

Training a pose LoRA on martial arts and dance references bakes the
reference subjects' faces into the LoRA. At generation time that LoRA
fights the character LoRA for control of the face, and the character
references lose. The pose LoRA should learn bodies and motion and know
nothing whatsoever about faces.

## Why masks, not gray ellipses

The obvious approach — blur the faces or paint a gray ellipse over them —
is worse than it looks, because whatever you put in those pixels *is a
training signal*:

- **Gray ellipse**: the LoRA learns "this concept comes with a gray blob"
  and pushes gray blobs into generations at higher weights.
- **Blur**: retains low-frequency structure — skin tone, hair colour,
  head shape, jaw silhouette — so identity leaks anyway, which is the
  whole thing being prevented.

**Loss masking** is strictly better: the face region contributes zero
gradient, so the model learns nothing there at all. No blob, no leak.

This is already supported by the trainer in use, with no modification.
Verified in `D:\Projects_26\AI_Image_Trainer\ai-toolkit-krea2`:

| Fact | Location |
|---|---|
| `mask_path`, `alpha_mask`, `mask_min_value` are dataset config keys | `toolkit/config_modules.py:955` |
| Masks matched by basename from a sibling folder | `toolkit/dataloader_mixins.py:1440` |
| Loss is multiplied by the mask: `loss = loss * mask_multiplier` | `extensions_built_in/sd_trainer/SDTrainer.py:870` |
| Masks receive the same bucket/crop/flip as their image | `toolkit/dataloader_mixins.py:1459-1495` |

Integration is one line in the dataset block:

```yaml
    datasets:
    - folder_path: D:/Projects_26/LoRA_Training_data/Poses/images
      mask_path:   D:/Projects_26/LoRA_Training_data/Poses/masks
      caption_ext: txt
```

### Contract

- **Faces are BLACK, everything else WHITE.** White is weight 1, i.e.
  trained. (`invert_mask` exists if the polarity is ever needed the other
  way round.)
- **Mask dimensions must equal the source image's**, at native size.
- **Same basename**, in the `masks/` folder: `punch_014.jpg` →
  `masks/punch_014.png`.
- `mask_min_value` defaults to `0.0` (fully ignored). If a hard zero ever
  produces boundary artefacts, `0.05`–`0.1` lets the region contribute a
  trace.

Redacted images are still produced, into `preview/`, but as a **review
aid** — a way to eyeball whether coverage is complete. The trainer
consumes the masks.

## Detection

```python
class Detector(Protocol):
    name: str
    def detect(self, image: np.ndarray) -> list[Box]: ...
```

Shipping implementation: **OpenCV YuNet** (`cv2.FaceDetectorYN`).
Confirmed present in both the project venv (OpenCV 4.11.0) and the system
install (5.0.0). No torch, no InsightFace compile pain on Python 3.12.

One dependency detail: the `face_detection_yunet_*.onnx` weights are
**not** shipped with the pip package and are not currently anywhere on
disk. ~350KB, vendored into the repo rather than downloaded at runtime so
the Olympus install script does not need another network fetch.

Fallback already available if YuNet disappoints: `kornia` is in the
dependency tree via ai-toolkit and ships `kornia.contrib.face_detection`.
The escalation path proper is a YOLO **head** detector via onnxruntime —
head detection beats face detection when subjects are turned away, which
is the common case in martial arts footage.

### Recall is the only metric that matters

A false positive costs a wasted region. A false negative puts an
unmasked face into training and defeats the feature. Martial arts and
dance are the hard case: heads turned away, extreme tilt, motion blur,
occlusion by a limb. Detectors *will* miss these.

Therefore the pipeline is **detect → review → export**, not
detect → export. The review pass is not optional polish; it is the part
that makes the feature trustworthy.

## Box expansion

Detectors return an eyes-to-chin box. Hair, hairline and jaw carry
identity too. Expansion is configurable, default ~1.6×, biased upward to
catch the hairline, clamped to image bounds.

Feathering is applied at mask generation — a few pixels of gradient at
the boundary, deliberately, in a mask that is otherwise hard-edged. Note
this interacts with the resampling rule in
[doc 03](03-dataset-model.md#mask-specific-resampling): feather at
generation, never acquire it accidentally through a resize.

## Sidecar state

Detected and hand-drawn boxes are persisted per image, so review work is
never lost and the pass is re-runnable:

```json
{
  "punch_014.jpg": {
    "boxes": [{"x": 412, "y": 88, "w": 96, "h": 128, "src": "yunet", "conf": 0.91},
              {"x": 640, "y": 120, "w": 88, "h": 110, "src": "manual"}],
    "reviewed": true
  }
}
```

Regenerating masks from this file is instant and does not re-run
detection. Changing the expansion factor re-renders every mask from
stored boxes rather than re-detecting.

## Review UI

A tab or mode over the gallery:

- Image with its boxes overlaid, mask preview toggleable.
- Draw a box (drag), delete a box (select + Delete).
- **Zero-detection images flagged loudly** and sortable to the front —
  these are where misses hide.
- Keyboard-first: next / previous / mark reviewed, so a few hundred
  images is a few minutes rather than an afternoon.
- Progress readout: `184/210 reviewed, 6 with no detections`.

## Pipeline

```
scan → detect (YuNet, threaded)
     → persist boxes to sidecar JSON
     → review (human, box editing)
     → export masks/*.png  + preview/*.jpg
     → validate (every image has a mask, dimensions match)
```

Export refuses to run, with a listing, if any image is unreviewed or has
zero boxes — overridable by an explicit flag for datasets where some
frames genuinely contain no face.
