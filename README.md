Port of https://github.com/dubhater/vapoursynth-adjust. Roughly same speed as akarin.Expr impl, both >1000 fps on 1080P YUV444 BlankClip.

## Usage

```python
core.adjust.Tweak(clip=clip, [float hue = 0, float sat = 1, float bright = 0, float  cont = 1, bool coring = True])
```
```
Parameters:

hue

Adjust the hue. Positive values shift it towards red, while negative values shift it towards green.

Range: -180.0 .. 180.0

sat

Adjust the saturation. Values above 1.0 increase it, while values below 1.0 decrease it. 0.0 removes all colour.

Range: 0.0 .. 10.0

bright

Adjust the brightness. This value is directly added to each luma pixel.

cont:

Adjust the contrast. Values above 1.0 increase it, while values below 1.0 decrease it.

Range: 0.0 .. 10.0
```

## Compilation
```bash
meson setup build
ninja -C build
ninja -C build install
```