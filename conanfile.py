from conans import ConanFile, CMake

class avg_slam_conan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"
    author = "ALT"

    options = {"pangolin": [True, False],
                "wsl": [True, False]}
    default_options = {"pangolin": False,
                        "wsl": False}

    def configure(self):
        self.options["opencv"].with_ffmpeg = False

    def requirements(self):
        self.requires("cmake/3.20.4")
        self.requires("eigen/3.3.9")
        self.requires("opencv/4.5.1")
        self.requires("ceres-solver/2.0.0")
        self.requires("yaml-cpp/0.6.3")
        self.requires("nlohmann_json/3.10.5")
        if self.options.pangolin:
            self.installPangolin()
            self.requires("pangolin/1.0@demo/testing")

    def imports(self):
        self.copy("*.dll", dst="bin", src="bin") # From bin to bin
        self.copy("*.dylib*", dst="bin", src="lib") # From lib to bin

    def build(self):
        cmake = CMake(self)
        cmake.definitions["pangolin_active"] = self.options.pangolin
        cmake.definitions["os_wsl"] = self.options.wsl
        cmake.configure()
        cmake.build()


    def installPangolin(self):
        self.run("cd pangolinSetup && conan export . demo/testing")
        self.run("cd pangolinSetup && conan install pangolin/1.0@demo/testing --build=missing")
