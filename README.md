# The T-Rex Selector -  The C++ Implementation

## Intro

## About the T-Rex Selector

### Dependencies

The following packages are required:

- **Eigen3** – Linear algebra library (any recent version)
- **Boost** ≥ 1.80 – Headers and iostreams components
- **Cereal** – Serialization framework  
- **OpenMP** – Parallel computing support
- **BLAS/LAPACK** – Linear algebra backend (Accelerate on macOS, OpenBLAS elsewhere)|

### Quick Install

**macOS:**

```bash
    brew install llvm eigen cereal boost openblas libomp
```

**Ubuntu/Debian:**

```bash
    apt install libeigen3-dev libcereal-dev libboost-all-dev libopenblas-dev libomp-dev
```

**Windows:**

```bash

```

## Notes

```txt

Step 1: Nuke the Build Cache
CMake heavily caches variables. If something goes wrong with compiler detection, you must delete the physical cache, not just "clean" the targets.

    Open the VSCode Explorer.

    Delete the entire build/ folder (which contains your debug/ and release/ preset folders).

    Delete the compile_commands.json file sitting at the root of your workspace (if it exists).


Step 2: Disarm Conflicting Extensions & Kits
Before reconfiguring, we must ensure VSCode isn't trying to hijack CMake.

    Extensions: Ensure conflicting linting extensions (like C/C++ Runner) are disabled. 
    You only want the official Microsoft C/C++ and CMake Tools extensions active.

    Kits: Look at the blue status bar at the bottom of VSCode. Find the CMake Kit section. 
    Click it and explicitly set it to [Unspecified]. 
    This forces VSCode to respect the Homebrew LLVM compiler block inside your CMakeLists.txt.


Step 3: Reconfigure the Project
Now we let CMake rebuild its cache and generate a fresh compilation database.

    Press Cmd + Shift + P (macOS) to open the Command Palette.

    Run the command: CMake: Delete Cache and Reconfigure.

    Alternative: If the CMake extension is being stubborn, you can run your custom task via 
    Terminal -> Run Task... -> CMake Full Build (Debug).

    Wait for the configuration to finish. 
    You should see your custom message *** Using Homebrew LLVM: ... in the output panel.


Step 4: Verify the Compile Commands
IntelliSense relies entirely on the compilation database to find your system headers.

    Look at the root of your VSCode Explorer tree.

    Verify that a fresh compile_commands.json file has automatically appeared there. 
    (This happens automatically because of the "cmake.copyCompileCommands": 
    "${workspaceFolder}/compile_commands.json" rule we added to your settings.json).


Step 5: Force an IntelliSense Rescan
Finally, we tell the C++ linter to wake up and read the new map.

    Press Cmd + Shift + P.

    Run the command: C/C++: Rescan Workspace.

    Give it a few seconds to parse. The red squiggly lines under your standard library includes 
    (like <vector> or <string>) will disappear.
```

When to use?

- After switching git branches where CMakeLists.txt might have changed.
- After installing a new system library via Homebrew (e.g., updating Boost or Eigen).
- If you ever see a red squiggly line under standard C++ headers.
- If CMake throws bizarre linker errors complaining about architectures (e.g., x86_x64 vs arm64).

## Development

- LLVM
- clangd
- VSCode Clang deactivated
- No Apple Clang compiler, no openMP support


```
cmake --install build/ --prefix /usr/local
```

````
/usr/local/
├── include/tsolvers/          ← all your .hpp files
├── lib/
│   ├── libtrex_tsolvers.a
│   └── cmake/TRexSelector/
│       ├── TRexSelectorConfig.cmake
│       ├── TRexSelectorConfigVersion.cmake
│       └── TRexSelectorTargets.cmake
```

A consumer then only needs:
```
find_package(TRexSelector 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE TRexSelector::tsolvers)
```
