# Near-pass-through output: cause and correction

The browser and `NeuralRenderer.run` initially defaulted local tone and structure to 0, with automatic masking disabled. The native demo defaults are local tone 1, local structure 1, skin structure -1, and automatic masking enabled (`upstream/OpenDLSS-NR/demo/nr_pass.h`, `NrControls`). The controls are network input features, not just final display adjustments.

The same 256 × 256 GTA V face crop, weights, seed 0, no temporal history, and unchanged kernels were used for every experiment. RGB differences below are measured after PNG-style 8-bit rounding against the resized input. The head measurement is before composition, over the source-sized RGB region.

| Tone | Structure | Automatic mask | Mean absolute head RGB | Mean RGB difference / 255 | Maximum RGB difference / 255 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | Off | 0.003464 | 0.047724 | 2 |
| 1 | 0 | Off | 0.023185 | 1.467651 | 7 |
| 0 | 1 | Off | 0.079815 | 5.076319 | 40 |
| 1 | 1 | Off | 0.103904 | 6.613439 | 38 |
| 0.5 | 0.5 | Off | 0.050303 | 3.185521 | 19 |
| 1 | 1 | On (native defaults) | 0.090447 | 5.755859 | 38 |

All measured head values were finite. At the old settings the 75 captured graph boundaries were also finite. Composition matches the native proxy-space expression: clamp(proxy + head.rgb / 4, 0, 1), then truncate to half. With no history, its temporal blend is zero. No residual multiplier or numerical kernel was changed to increase the effect.

The corrected browser run completed in approximately 5 seconds. Omitted API conditioning was checked bit for bit against explicitly supplying native defaults. The browser UI test now requires tone=1, structure=1, and automatic masking on. Explicit zeros remain supported. Fixture replay retains its prior zero fallbacks rather than adopting interactive defaults.

The source image, cropped examples, and local diagnostic captures are intentionally excluded from the public repository. This report records measurements from local development.

This explains the near-pass-through result in the tested images. It does not establish whole-network equivalence to a capture from NVIDIA's native implementation.

Screenshot attribution: GTA V / Rockstar Games, image obtained from [SVG](https://www.svg.com/387581/the-nicest-person-to-beat-gta-5-is-still-a-monster/). Original image: `https://www.svg.com/img/gallery/the-nicest-person-to-beat-gta-5-is-still-a-monster/l-intro-1618930071.jpg`. Crop: x=300, y=0, width=900, height=900, resized to 256 × 256. Source screenshot modification history is unknown. Example game imagery is not covered by this project's code license.
