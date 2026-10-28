# HRIR data

The head-related impulse responses in this directory are derived from the
**SADIE II database**, subject D2 (KEMAR, head and torso; 8802 directions
over the whole sphere, 48 kHz, diffuse-field equalized):

> Copyright 2018, University of York. Licensed under the Apache License,
> Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
>
> C. Armstrong, L. Thresh, D. Murphy, G. Kearney, "A Perceptual Spectral
> Difference Model for Binaural Signals", Applied Sciences 8(11):2029
> (2018), https://doi.org/10.3390/app8112029.
> Database: https://www.york.ac.uk/sadie-project/database.html

Modified by `tools/make_hrir_sofa.py`: each ear made minimum phase with
the measured interaural delay restored, left/right symmetrized, truncated
to 128 taps, the grid thinned to ~3° along each measured ring, scaled to
the engine's previous diffuse-field energy.
