# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/workspace/LightTorch/LightTorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/workspace/LightTorch/LightTorch/build

# Include any dependencies generated for this target.
include CMakeFiles/ndarray_backend_cpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ndarray_backend_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ndarray_backend_cpu.dir/flags.make

CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o: CMakeFiles/ndarray_backend_cpu.dir/flags.make
CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o: ../src/ndarray_backend_cpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/LightTorch/LightTorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o -c /root/workspace/LightTorch/LightTorch/src/ndarray_backend_cpu.cc

CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/LightTorch/LightTorch/src/ndarray_backend_cpu.cc > CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.i

CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/LightTorch/LightTorch/src/ndarray_backend_cpu.cc -o CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.s

# Object files for target ndarray_backend_cpu
ndarray_backend_cpu_OBJECTS = \
"CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o"

# External object files for target ndarray_backend_cpu
ndarray_backend_cpu_EXTERNAL_OBJECTS =

../python/ltorch/backend_ndarray/ndarray_backend_cpu.cpython-39-x86_64-linux-gnu.so: CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
../python/ltorch/backend_ndarray/ndarray_backend_cpu.cpython-39-x86_64-linux-gnu.so: CMakeFiles/ndarray_backend_cpu.dir/build.make
../python/ltorch/backend_ndarray/ndarray_backend_cpu.cpython-39-x86_64-linux-gnu.so: CMakeFiles/ndarray_backend_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/LightTorch/LightTorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../python/ltorch/backend_ndarray/ndarray_backend_cpu.cpython-39-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ndarray_backend_cpu.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /root/workspace/LightTorch/LightTorch/python/ltorch/backend_ndarray/ndarray_backend_cpu.cpython-39-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/ndarray_backend_cpu.dir/build: ../python/ltorch/backend_ndarray/ndarray_backend_cpu.cpython-39-x86_64-linux-gnu.so

.PHONY : CMakeFiles/ndarray_backend_cpu.dir/build

CMakeFiles/ndarray_backend_cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ndarray_backend_cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ndarray_backend_cpu.dir/clean

CMakeFiles/ndarray_backend_cpu.dir/depend:
	cd /root/workspace/LightTorch/LightTorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/LightTorch/LightTorch /root/workspace/LightTorch/LightTorch /root/workspace/LightTorch/LightTorch/build /root/workspace/LightTorch/LightTorch/build /root/workspace/LightTorch/LightTorch/build/CMakeFiles/ndarray_backend_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ndarray_backend_cpu.dir/depend

