# Change Log:

## Table of Contents

- [Change Log:](#change-log)
	- [Table of Contents](#table-of-contents)
		- [Version 1.0.0 (2022-12-27)](#version-100-2022-12-27)
		- [Version 1.0.1](#version-101)
		- [Version 1.1.0 (in progress)](#version-110-in-progress)

### Version 1.0.0 (2022-12-27)
1. iterate (repeatedly) over each mouse and each week (each mouse and each experiment)
    - [x] get project files (experimental) structure
    - [x] initialize an iterator over the project structure
2. get daq data from CSV file
    - [x] read CSV files
    - [x] save each column from CSV file
        * Note: CSV columns are of differing lengths
3. get “beh_metadata” from json metadata
    - [x] read JSON file
    - [x] grab the values for key “beh_metadata”
        - [x] get the values of sub key “trialArray”
        - [x] get the values of sub-key “ITIArray”
4. get video metadata from *.mp4 file (with ffmpeg.probe)
    - [x] read in the *.mp4 metadata
    - [x] select the correct video stream
    - [x] get the average frames per second
5. get SLEAP data from *.h5 file
    - [x] open h5 file
    - [x] get transposed values of key “tracks” (tracking_locations)
    - [x] fill missing locations (linear regress. fit)
    - [x] get transposed values of key “edge_inds”
    - [x] get values of key “edge_names”
    - [x] get transposed values of “instance_scores”
    - [x] get transposed values of “point_scores”
    - [x] get values of “track_occupancy”
    - [x] get transposed values of “tracking_scores”
    - [x] get decoded values of “node_names” (make sure there's no encoding issues)
6. deconstruct SLEAP points into x and y points (across all frames)
    - [x] iterate over each node
    - [x] breakup the 4D array “tracks” into 1D array for x and y values respectively
        * Note: [frame, node, x/y, color] for greyscale the color dimension is 1D (i.e. essentially the 4D array is 3D because the color dimension is constant)
    - [x] iterate over each frame
    - [x] assign mouse, week, frame #, and timestamp (using average frames per second)
7. Split data into individual trials by trial type using the Speaker and LED data from the CSV daq data
    - [x] initialize trial iterators for the consistently documented points from the daq CSV
    - [x] iterate over each trial in “trialArray”
    - [x] get the index of 10sec before and 13sec after trial start
    - [x] for each feature, grab the start and end indices
    - [x] store data from each trial in a pd.dataframe
    - [x] concatenate all pd.dataframes together for each video
    - [x] concatenate the pd.dataframes from each video together for each mouse (base expr split)
8. Prepare the data
    - [x] (opt.) mean center across all points for a single trial
    - [x] mean center across all trials for a single experiment
    - [x] mean center across all experiments for a single mouse
    - [x] mean center across all mice
    - [x] (opt.) z-score mean-centered data
9. Analyze the data
    - [x] Perform 2D and 3D PCAs on all data (raw, centered, by trial, by week, by mouse, etc…)
    - [x] apply gaussian kernel to PCA outputs
10. Save the data
    - [x] write everything to HDF5 file(s)
### Version 1.0.1
  - [x] add exhaustive documentation
  - [x] add inline documentation
  - [x] strengthen type hints
  - [x] Fix bug where the `CustomColumn` class is not properly initialized
  - [x] Fix bug where the `CustomColumn` class is not properly built
  - [x] Fix bug where the `CustomColumn` class is not properly appended
  - [x] Fix bug where the `trials` and `trialData` attributes were not properly initialized
  - [x] Fix bug where the `trials` and `trialData` attributes were not properly built
  - [x] Fix bug where the `meanCenter` did not properly mean center the data recursively
### Version 1.1.0 (in progress)
  - [x] add support for multiple mice
  - [ ] add clustering/prediction algorithm(s)
  - [ ] add velocity, acceleration, and jerk calculations
  - [x] add save option for all data
  - [x] add plotting functions
  - [ ] clustering features
      - [ ] distance to a point
          - [ ] vector to a point (theta, magnitude) or (angle, distance)
      - [ ] velocity/acceleration
      - [ ] distance to centroid
      - [ ] distance between given points
