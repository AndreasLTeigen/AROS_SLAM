This repository uses the package manager Conan.  

To install Conan run the command:  
    $pip install conan

This means that all required packages can be installed with the command:  
    $conan install /path/to/folder/with/conanfile/ -if /path/to/build/folder/ --build=missing  

If your present working directory is the project folder with the conanfile.py, simply type:  
    $conan install . -if build/ --build=missing  

    Install options:
        - pangolin      (Includes a 3D view of trajectory and pointcloud, default value=False)
        - wsl           (Use this option if installing on Windows Subsystems for Linux, default value=False)
    Install options are included with the marker '-o' then the option name and the value.
    For instance '-o pangolin=True'. All options has to use the '-o' prefix.

Go to the main.cpp file and change the path to a path containing the image series you want to play.  

To build the code type:  
    $conan build /path/to/folder/with/conanfile/ -bf /path/to/build/folder/

If your present working directory is the project folder with the conanfile.py, then type:  
    $conan build . -bf build  

To run the code use the command:  
    $./build/bin/AVG_SLAM




List of methods that are implemented:

    Motion prior:
        - "constant":   Expect acceleration from previous frame to current frame to be within a Gaussian profile.
        - "gt":         Retrieves ground truth pose information.

    Keypoint exctraction:
        - "orb":        Oriented fast, rotated brief detection.
        - "orb_nb":     Orb + naive bucketing.
        - "orb_gt_nb":  Orb + naive bucketing + prioritize reprojected points from previous frame based on GT.
    
    Keypoint matching:
        - "bf_mono":    Brute force matching based only on descriptors. Uses Lowes ratio test, selects only one match candidate per point.

    Pose calculation:
        - "5-point + outlier removal":  Uses 5-point algorithm to calculate pose between frames which include a RANSAC outlier removal process.
        - "motion prior":               Simply returns the motion prior.
    
    Map point registration:
        - "all":        Registers or updates all map points in the newest frame that was matched with the previous frame.
        - "depth_gt":   Registers points from the newest frame that has not been observed before based on a ground truth depth map.

    Map point culling:
        - "OoW":        Removes all map points no longer visible by any frame in a temporal frame window.
    
