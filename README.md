# bandpass-stability

The aim of these tools is to gauge the spectral and temporal stability of a radio interferometer. 
Please read this page prior to use, as there are some important caveats, particularly with respect
to reference antennas.

## Installation

Clone this repo, create a fresh python virtual environment and you can `pip install` it. 
Using `pip install -e` is recommended, as the project should be considered to be under
active development and as always some bugs will probably be uncovered as it gets used
on more results.


## Method

The input product for these tools is a set of CASA-format bandpass tables. Support for 
other formats can be added as required. Each table will at present be treated as a single 
epoch, with stability metrics being computed by analysing the properties of the 
bandpass over the whole set. Intra-table processing for very short timescale stability 
analyses is also pending, and will likely be added as a separate command line utility.

The workflow and each associated tool is as follows:

1. `bandpass-smooth`: A perfectly stable instrument will still exhibit 
    apparent temporal variations in its bandpass response due to the unavoidable
    effects of thermal noise. To minimise the contribution of noise to the stability
    metrics it is advantageous to first de-noise the bandpass solutions along the 
    frequency axis using a median filter with a width chosen appropriately such that
    it does not absorb genuine bandpass structure. This is particularly important for
    high frequency resolution bandpass solutions.
2. `bandpass-map`: The bandpass table only contains antenna indices and there is a risk
    that these could change between successive bandpass calibration scans should a different 
    sub array be used or an antenna becomes unavailable. This script will construct a CSV file
    that contains a mapping of antenna indices to physical antennas for the set of bandpass
    calibration tables under scrutiny. There is a hardwired assumption that the name of each 
    bandpass table contains within it the <i>complete</i> directory name of the parent MS. 
3. `bandpass-lag`: Finally, this script ingests the CSV antenna mapping as well as the (smoothed)
    bandpass tables. Each pair of time-adjacent tables is compared, and plots of per-pair stability
    metrics are produced for each antenna as a function of frequency and time, as well as a CSV
    dump of the relevant stats.

Each tool has a minimal CLI which can be viewed using the `-h` command line switch.

The outputs are written for each antenna into `lag_residual_results/<antenna_name>/`. Each subdirectory contains:

- `<antenna>_pair_metrics.csv`: One row per adjacent table pair, with
  amplitude and phase RMS residuals over all common channels.
- `<antenna>_pair_rms.png`: The aforementioned RMS values plotted as a function of time. A
  sudden jump in this plot indicating a sudden change in the instrumental
  bandpass over that interval.
- `<antenna>_channel_summary.png`: The median amplitude and phase residual (16th / 84th percentile range)
  per channel, calculated across all adjacent
  pairs. This plot shows which channels are most/least stable over time, as well as any overall frequency 
  dependence in the bandpass stability.


## Metrics

### Per-pair stability metric

For each adjacent pair of bandpass tables ${B_n}(\nu)$ and ${B_{n+1}}(\nu)$ the lagged complex ratio $R_n(\nu)$ is formed:

$$
R_n(\nu) = \frac{B_n+1(\nu)}{B_{n}(\nu)}
$$

The log-amplitude and phase residuals are then:

$$
A_n(\nu) = \log|R_n(\nu)|, \qquad
\qquad \phi_n(\nu) = \arg\big(R_n(\nu)\big).
$$

The RMS of $A_n(\nu)$ and $\phi_n(\nu)$ are then determined over all common channels for a given pair:

$$
\mathrm{RMS}_A,n = \sqrt{\big\langle A_n(\nu)^2 \big\rangle_\nu}, \qquad
\mathrm{RMS}_\phi,n = \sqrt{\big\langle \phi_n(\nu)^2 \big\rangle_\nu}.
$$

These values are written to the per-antenna `pair_metrics.csv` files under the 
`amplitude_rms_percent_approx` and `phase_rms_degrees` columns, as well as plotted
in the `<antenna>_pair_rms.png` as a function of time / table pair.

### Per-channel overview

Across all adjacent pairs for one antenna (i.e. over the pair index $n$), the median and 16th/84th percentile spread of $A_n(\nu)$ and $\phi_n(\nu)$ is determined, independently for each channel:

$$
\tilde{A}(\nu) = \mathrm{median}_{\,n}\big[A_n(\nu)\big], \qquad
\tilde{\phi}(\nu) = \mathrm{median}_{\,n}\big[\phi_n(\nu)\big]
$$

These quantities are plotted on the two panels of `<antenna>_channel_summary.png`, with the solid line being the median and the shaded region the 16th–84th percentile range.


## Radio frequency interference

Strong (residual) radio interference present in the visibilities will also corrupt the corresponding
bandpass solutions. This will then egregiously affect the sensitivity metrics, introducing
strong deviations from epoch to epoch that is unrelated to the stability of the actual telescope.
This will generally affect a limited region of the band and is thus easily identifable in the per-channel
overview plots, however despite that it can still adversely affect the per-pair stability metrics,
giving the impression of significant epoch to epoch variability. 

While it is assumed that radio frequency interference (RFI) has been flagged as standard prior to
the creation of the input set of bandpass tables, experience shows that strong residuals can remain,
particularly close to regions of high RFI occupancy such as those caused by the geolocation 
satellites in L-band.

In an attempt to guard against this, channels affected by transient RFI are identified in $A_n(\nu)$ and $\phi_n(\nu)$
using an iterative clipping method. The median and median absolute deviation (MAD) are computed from the unflagged channels
and any channel that deviates from the median by more than a chosen multiple of the MAD-related scatter is
flagged. The median and MAD are recomputed from the surviving channels and the process repeats until an 
iteration occurs where zero channels are dropped.

The `<antenna>_pair_metrics.csv` file contains `amplitude_channels_clipped` and `phase_channels_clipped` columns, 
which are the counts of the channels excluded when calculating the statistics for a given pair. It is 
advisable to examine these numbers (along with the general flag statistics of the observation that was
used to generate the bandpass table) prior to drawing any conclusions. 


## Reference antennas 

Bandpass solutions are generally solved for with respect to a chosen 'reference
antenna' to break degeneracies in the process. The phases for this antenna are fixed to zero, 
and all other phase corrections are given with respect to it. The most obvious upshot of this
is that for the chosen reference antenna the phase stability will simply be consistent with floating
point noise, at around $10^{-7}$ degrees.

The second and less obvious effect is that a sudden change in the phase stability of the reference
antenna will propagate coherently into all of the other antennas in the array. Thus if the set of
`<antenna>_pair_rms.png` images exhibit a sudden jump that is common to all antennas the prime
suspect is likely to be a sudden change in the behaviour of the reference antenna.

In order to produce stability metrics for the reference antenna itself a set of bandpass tables
that utilise an alternative (known to be good) reference antenna should be produced. These will 
of course be subject to the same set of caveats.

There is also a possible scenario whereby a set of bandpass solutions are being produced for a
given antenna which unexpectedly fails or is compromised for some of the input scans. When this
happens some software will switch to a different antenna and continue the process. This generally
has no effect on the quality of the calibration, simply introducing a different zero-point for 
the relative phases at a certain time. It does however introduce discontinuities that will
adversely affect a stability analysis such as the one deployed here. Users should ensure that the
set of bandpass tables that are fed to these tools have a consistently good reference antenna
throughout.

Note that amplitude stability metrics are not affected by reference antenna issues.