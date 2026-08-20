# 03 — dataset model

## The invariant

A training example is not a file. It is a **bundle**:

```
punch_014.jpg          the image
punch_014.txt          the caption
masks/punch_014.png    the loss mask   (optional)
```

Every operation that touches one member must touch all of them, or the
bundle silently desynchronises. v1 has no such concept — `fix_images`,
`create_duplicates` and `mass_rename_images` each hand-roll their own
sidecar handling (`_copy_text_file`, `_copy_transformed_text_file`, and an
inline block in the rename loop). Three copies, three chances to forget.

Adding masks to that structure means three more chances to forget, and
the failure is **silent**: ai-toolkit sets `has_mask_image = True` only if
it finds a basename match, otherwise it trains the image unmasked without
warning (`toolkit/dataloader_mixins.py:1445`). A desynchronised bundle
means a face trains that was supposed to be excluded — exactly the thing
the feature exists to prevent.

## `DatasetItem`

```python
@dataclass
class DatasetItem:
    image: Path
    caption: Path | None
    mask: Path | None
    quality: str | None          # good / ok / bad, from the captioner

    @property
    def stem(self) -> str: ...
    def sidecars(self) -> Iterator[Path]: ...
    def renamed_to(self, new_stem: str) -> "DatasetItem": ...
```

Rules:

- **Nothing outside `dataset/` opens a folder and globs for images.** One
  scanner, one extension list, one definition. v1 has two copies of the
  extension tuple in two classes, which is how the gallery and the
  processor can disagree about what exists.
- **Operations take and return `DatasetItem`s.** A rename is a pure
  mapping from old items to new items; execution is separate from
  planning.
- **Sidecars follow automatically.** `sidecars()` is the single place
  that knows a bundle has a caption and a mask. Add a fourth member later
  — control images, depth maps — and every operation gets it for free.

## Operations

Ported from the v1 methods, with the fixes already made in `d1890ce`
carried forward rather than re-derived.

| Operation | v1 origin | Changes |
|---|---|---|
| `scan` | `get_image_files` ×2 | One implementation. Wider extension list. Returns items, not filenames. |
| `resize` | `fix_images` | EXIF baked in; handles released; masks resized with nearest-neighbour, never Lanczos. |
| `rename` | `mass_rename_images` | Plan/execute split; two-phase with rollback; masks renamed with the bundle. |
| `augment` | `create_duplicates` | **Masks transformed identically to their image.** A flipped image needs a flipped mask. |
| `mask` | new | See [04 — face masking](04-face-masking.md). |
| `caption` | `generate_captions_claude` | Ported; writes through `DatasetItem`. |

### Mask-specific resampling

The one genuinely new correctness rule. A mask is not a photograph:

- Resize with `NEAREST`, or with `LANCZOS` followed by a re-threshold.
  Smooth interpolation on a hard-edged mask produces grey fringes, and a
  grey pixel is a *partial* loss weight — a soft leak of exactly the
  region being excluded.
- Feathering, where wanted, is applied deliberately at generation time
  (see doc 04), not acquired accidentally through resampling.
- Masks are written as 8-bit greyscale PNG, never JPEG. JPEG ringing on a
  hard black/white edge produces the same grey-fringe problem.

## Caption storage

v1 stores captions in `.txt` sidecars *and* in a JSON file, with
precedence resolved case by case in
`DataManager._get_description_for_image`. Two sources of truth that drift.

**v2: `.txt` sidecars are the truth.** They are what every trainer
actually reads — `caption_ext: txt` in the Krea2 configs. The JSON
becomes a pure cache of derived metadata (quality ratings, detected face
boxes, review state), rebuildable from scratch, never authoritative for
caption text.

## Validation

A `validate()` pass that reports, without modifying anything:

- images with no caption, or an empty one
- captions with no image (orphans from a broken rename)
- **masks with no image, and images with no mask when masking is enabled**
- mask dimensions not matching their image — ai-toolkit warns and tries to
  swap sizes (`dataloader_mixins.py:1473`); better to catch it here
- images below the resolution floor
- duplicate or near-duplicate stems

This is the thing that would have caught every v1 dataset bug before it
reached a training run, and it is cheap.
