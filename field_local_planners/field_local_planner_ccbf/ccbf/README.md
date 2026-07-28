# ccbf

`ccbf` is a small C++14 control-barrier-function safety filter for a planar
omnidirectional robot. It uses Eigen for linear algebra and the dense ProxQP
solver provided by ProxSuite.

The package supports two build modes:

- a standalone CMake build from this directory, without Catkin;
- the existing Catkin build as part of the parent workspace.

## Dependencies

The standalone build requires:

- CMake 3.10 or newer;
- a C++14 compiler;
- Eigen3;
- ProxSuite;
- GoogleTest when `BUILD_TESTING=ON`.
- Python 3 and Matplotlib for plotting the simulation.

If ProxSuite is installed under a non-standard prefix such as
`/opt/openrobots`, append that prefix to `CMAKE_PREFIX_PATH` or pass it during
configuration:

```bash
-DCMAKE_PREFIX_PATH=/opt/openrobots
```

## Standalone build

Run these commands from the `ccbf` directory:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build --parallel
```

If ProxSuite is not found automatically:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/opt/openrobots
cmake --build build --parallel
```

### Run the example

```bash
./build/ccbf_demo
```

The example filters the nominal command `(1, 0, 0)` near a circular obstacle.
It should report a safe command close to `(0.75, 0, 0)` and finish with
`CCBF demo passed`.

### Run and plot the five-obstacle simulation

Generate a CSV containing the robot trajectory, nominal and safe commands,
minimum barrier value, goal, and all five obstacle definitions:

```bash
./build/ccbf_simulation build/ccbf_simulation.csv
```

Open the Matplotlib visualization:

```bash
python3 examples/plot_simulation.py build/ccbf_simulation.csv
```

The plot shows the straight nominal pathline, the CBF-filtered trajectory,
the five safety circles, velocity filtering, and the closest barrier value
over time.

To save the plot without opening a window:

```bash
python3 examples/plot_simulation.py \
  build/ccbf_simulation.csv \
  --save build/ccbf_simulation.png \
  --no-show
```

### Run the tests

Run the complete standalone test suite:

```bash
ctest --test-dir build --output-on-failure
```

Run the unit-test executable directly:

```bash
./build/ccbf_test
```

### Install the standalone package

```bash
cmake --install build --prefix install
```

This installs the library, headers, example executable, and a CMake package
configuration. Downstream CMake projects can then use:

```cmake
find_package(ccbf REQUIRED)
target_link_libraries(my_target PRIVATE ccbf::ccbf)
```

Configure the downstream project with
`-DCMAKE_PREFIX_PATH=/path/to/ccbf/install` when necessary.

## Catkin build

From the Catkin workspace root:

```bash
catkin build ccbf
source devel/setup.bash
```

Run the example:

```bash
rosrun ccbf ccbf_demo
```

Generate simulation data:

```bash
rosrun ccbf ccbf_simulation /tmp/ccbf_simulation.csv
```

Plot it from the `ccbf` source directory:

```bash
python3 examples/plot_simulation.py /tmp/ccbf_simulation.csv
```

Run the Catkin tests and display their result:

```bash
catkin run_tests ccbf
catkin_test_results build/ccbf
```
