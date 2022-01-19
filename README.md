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