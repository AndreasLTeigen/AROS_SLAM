This repository uses the package manager Conan.  

To install Conan run the command:  
    $pip install conan

This means that all required packages can be installed with the command:  
    $conan install /path/to/folder/with/conanfile/ -if /path/to/build/folder/ --build=missing  

If your present working directory is the project folder with the conanfile.py, simply type:  
    $conan install . -if build/ --build=missing  

Go to the main.cpp file and change the path to a path containing the image series you want to play.  

To build the code type:  
    $conan build /path/to/folder/with/conanfile/ -bf /path/to/build/folder/

If your present working directory is the project folder with the conanfile.py, then type:  
    $conan build . -bf build  

To run the code use the command:  
    $./build/bin/AVG_SLAM