# NWS Flood Stage Categories

The U.S. National Weather Service defines flooding at a river gauge using a set of
site-specific stages, calibrated to local channel geometry and historical impact
data rather than a single universal numeric threshold:

- **Action stage**: the gauge height at which some mitigation action (e.g. notifying
  emergency management, closing low-lying roads) should begin. Not yet flooding.
- **Flood stage (minor)**: water begins to affect low-lying areas, minor road closures,
  minimal or no property damage.
- **Moderate flood stage**: some inundation of structures and roads near the river,
  evacuations of some homes/businesses may be required.
- **Major flood stage**: extensive inundation of structures and roads, significant
  evacuations of people and property.

Because stage thresholds are site-specific (a function of channel geometry, floodplain
development, and levee/infrastructure protection), a raw gauge-height or discharge
reading cannot be classified as "flood" or "no flood" without reference to the specific
site's rating curve and locally-defined stage thresholds. This is a key reason a
classification agent should ground numeric readings against site-specific reference
data (or, in its absence, relative deviation from the site's historical baseline) rather
than a single global numeric cutoff.

# WMO Definition of a Flood

The World Meteorological Organization defines a flood as "a rise, usually brief, in the
water level in a stream to a peak from which the water level recedes at a slower rate."
More broadly, floods are described as the temporary covering of normally dry land by
water outside its usual confines, arising from riverine overflow, storm surge/coastal
inundation, or intense local rainfall exceeding drainage capacity (pluvial/flash flooding).

# Flood Types Relevant to This Benchmark

- **Riverine flooding**: sustained or peak river discharge/stage exceeds the channel's
  capacity, typically from prolonged or intense upstream precipitation. Indicated by a
  sustained rise in discharge (cfs) and gauge height (ft) over hours to days.
- **Flash flooding**: rapid-onset flooding, typically within 6 hours of the causative
  rainfall, in small/steep watersheds or urban areas with limited drainage capacity.
  Indicated by a very sharp, short-duration spike in discharge.
- **Storm-surge / coastal flooding**: wind-driven sea-level rise during tropical or
  extratropical cyclones, which can also back up tidal rivers and elevate gauge
  readings far upstream from the coast (as seen with Hurricane Sandy on NJ's tidal
  rivers).
