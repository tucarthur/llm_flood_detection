# Mineirinho Creek Water-Level Classification Criteria (this benchmark's label taxonomy)

This benchmark classifies fixed-camera images of Mineirinho Creek (Sao Carlos, Brazil,
part of the E-Noe monitoring project) into four ordinal water-level categories:
**low < medium < high < flood**. The camera looks down into a concrete drainage canal
from a road bridge; the canal has a walled concrete channel on both banks, with a grass
embankment above the wall on the far (camera-facing) side.

## The four categories

- **low**: normal/dry-weather water level. A large, clearly visible band of concrete
  wall is exposed between the water surface and the top of the wall/grass line. This is
  the overwhelming majority case (~99% of images) -- the creek is described as "a very
  shallow stream whenever the weather is dry."
- **medium**: water has risen but a visible margin of concrete wall remains between the
  water and the grass/wall-top.
- **high**: water has risen further, with little or no concrete wall margin remaining,
  but has **not yet** crossed onto the grass bank.
- **flood**: water has risen **above the top of the wall and is touching the grass
  bank**. This is a specific physical threshold, not a fuzzy "very severe" judgment --
  the defining visual test is whether water is in contact with vegetation/grass above
  the wall line, not how much water is visible or how brown/turbid it is.

## Reference marker

The original annotators used a physical reference marker/pole (visible in many frames,
roughly mid-frame near the wall, sometimes with colored bands) already deployed at the
canal to distinguish low from medium and medium from high. If a marker is visible in an
image, use its position relative to the waterline as a calibration reference alongside
the wall/grass criterion above.

## Severe class imbalance -- calibrate your prior accordingly

Across the labeled dataset: low=98.84%, medium=0.78%, high=0.27%, flood=0.11% of images.
The base rate for anything other than "low" is under 1.2%. Do not let this bias you
toward under-calling `flood` when the wall/grass criterion is actually met -- the
imbalance reflects how rare flooding is at this site, not how hard the flood category is
to recognize when it occurs. A missed `flood` (false negative) is the costliest error in
this benchmark's evaluation.

## Nighttime images are a major visual challenge -- flag your confidence accordingly

Roughly half of all images are nighttime, and severe-category images skew disproportionately
night-heavy (64% of `flood` images, 55% of `high` images are nighttime, vs. 47% for `low`)
-- flash floods here tend to occur during nighttime storms. Nighttime frames are often
dominated by a single overexposed streetlight glare near the top-center of the frame,
which can wash out most of the canal and make the waterline very difficult to judge
directly. When classifying a nighttime image with heavy glare, say so explicitly in your
rationale and lower your confidence accordingly rather than guessing with false certainty.
