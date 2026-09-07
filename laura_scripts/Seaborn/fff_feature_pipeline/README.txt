FFF FEATURE PIPELINE
====================

Put these files together in one folder in your PyCharm project.
Run build_fff_dataset.py for ONE recording at a time.

Edit only the configuration section at the top of build_fff_dataset.py:
- recording root / overview path
- contrast-step stimulus ID
- SCF stimulus ID
- chirp stimulus ID
- moving-bar stimulus ID and timing parameters
- zero or more moving-edge conditions, each with its own H5 path

Outputs:
    <recording>/fff_analysis/fff_features.csv
    <recording>/fff_analysis/fff_data.nc

The NetCDF is indexed by cell_index and can be reloaded with:

    import xarray as xr
    fff = xr.load_dataset(r"...\fff_analysis\fff_data.nc")

Main metrics
------------
Contrast steps:
- csteps_polarity_index
- csteps_on/off_transient_hz
- csteps_on/off_sustained_hz
- csteps_on/off_transience_index
- csteps_compound_transience_index
- per-contrast response amplitudes

SCF:
- per-wavelength ON response
- per-wavelength OFF response
- per-wavelength signed tuning (ON - OFF)
- per-wavelength amplitude
- preferred wavelength
- colour-opponent flag
- opponency strength
- zero crossings

Chirp:
- existing preferred/max-power frequency
- existing frequency threshold
- existing power metrics

Moving bar:
- DSI (peak-versus-opposite; matches supplied summary figure)
- vector DSI
- OSI
- preferred direction/orientation
- eight directional responses

Moving edge (each condition kept separate):
- DSI (peak-versus-opposite; matches supplied edge figure)
- vector DSI
- OSI
- preferred direction/orientation
- eight directional responses

Important distinction:
Moving bar and moving edge are intentionally independent analyses. Moving bar uses
fixed frames-per-direction timing from the summary plot. Moving edge reads exact
Direction_Start_Frame / Direction_Frame_Count / Frame_Rate metadata from each H5
and applies the supplied projector direction transformation.
