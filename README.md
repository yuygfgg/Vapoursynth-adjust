Port of http://avisynth.nl/index.php/Tweak. `interp` `dither` `realcalc` `dither_strength` are not currently supported.

## Usage

```python
core.adjust.Tweak(clip=clip, [float hue = 0.0, float sat = 1.0, float bright = 0.0, float cont = 1.0, bool coring = True, float startHue = 0.0, float endHue = 360.0, float maxSat = 150.0, float minSat = 0.0])
```

Also see http://avisynth.nl/index.php/Tweak and https://github.com/dubhater/vapoursynth-adjust.

## Compilation
```bash
meson setup build
ninja -C build
ninja -C build install
```