# snap_conv
A code for universal reading and writing of snapshots from astrophysical simulations.

This code is made to make handling astrophysical snapshots easier by providing a universal interface for interaction.

## Getting Started
Loading the data is simple.
Let's say you have a snapshot from the SWIFT code named "snap_0090.hdf5"
```py
data = snap_conv.SwiftFrontend("./snap_0090.hdf5")
```
Accessing field from the dataset is simple!
```py
# get the gas coordinates
data.gas.Coordinates
# get the dark matter masses
data.dark_matter.Masses
```
Where relevant, quantities are represented using arrays from unyt.
No more forgetting factors of h!

Fields are lazily loaded and cached so subsequent references to a field does not need to wait to read the file again.
Datasets act as a least-recently-used cache (the size of which can be set by the `cache_size` parameter) and will drop previously loaded fields when loading new ones, keeping memory usage below the set threshold.
Note that the dataset will drop _its_ reference to older fields. Any references you take will still be valid.
This means the data will only release its memory if you do not have any references of your own holding on to it.

## Writing
To convert from one snapshot format to another is a single function call.
```py
data = snap_conv.SwiftFrontend("./snap_0090.hdf5")
data.write_as(snap_conv.GadgetFrontend, "converted.hdf5")
```

## TODO
- [ ] Writing SWIFT snapshots.
- [ ] Universal `snap_conv.load` function which detects file type.
- [ ] Executable for converting snapshots from the shell.
