from conans import ConanFile, CMake, tools


class PangolinConan(ConanFile):
    name = "pangolin"
    version = "1.0"
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of Pangolin here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    generators = "cmake"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def source(self):
        self.run("git clone --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git")
        # This small hack might be useful to guarantee proper /MT /MD linkage
        # in MSVC if the packaged project doesn't have variables to set it
        # properly
        tools.replace_in_file('Pangolin/CMakeLists.txt', 'project("Pangolin")',
                              '''project("Pangolin")
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()''')


    def requirements(self):
        self.requires("cmake/3.20.4")
        self.requires("glew/2.2.0")
        self.requires("eigen/3.3.9")
        self.requires("opengl/system")
        self.requires("wayland/1.19.0")
        self.requires("pybind11/2.6.2")
        self.requires("pkgconf/1.7.4")

        #LATE NIGHT ADDINGS
        self.requires("egl/system")
        self.requires("libjpeg/9d")
        self.requires("lz4/1.9.3")
        self.requires("libpng/1.6.37")

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder="Pangolin")
        cmake.build()

        # Explicit way:
        # self.run('cmake %s/hello %s'
        #          % (self.source_folder, cmake.command_line))
        # self.run("cmake --build . %s" % cmake.build_config)

    def package(self):
        self.copy("*.h", dst="include/", src="Pangolin/include/")
        self.copy("*.hpp", dst="include/", src="Pangolin/include/")
        self.copy("*.h", src="src/include/", dst="include/")
        self.copy("*.hpp", src="src/include/", dst="include/")
        #self.copy("*hello.lib", dst="lib", keep_path=False)
        #self.copy("*.dll", dst="bin", keep_path=False)
        #self.copy("*.so", dst="lib", keep_path=False)
        #self.copy("*.dylib", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["pangolin"]

