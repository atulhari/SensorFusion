# SplineFusion Refactoring

This document describes the refactoring of the SplineFusion codebase to separate the core algorithm from ROS dependencies.

## Key Changes

1. **Created ROS-Independent Core Library**
   - Implemented a clean API in `SplineFusionCore.hpp`
   - Separated algorithm logic from ROS message handling
   - Created well-defined data types in `Types.hpp`
   - Modularized the algorithm into components:
     - `WindowManager`: Handles the sliding window of control points
     - `Optimizer`: Manages the optimization process
     - `DataProcessor`: Handles IMU integration and data preparation
     - `SplineState`: Manages the B-spline representation

2. **Improved Interface Design**
   - Created a cleaner interface with proper data encapsulation
   - Removed global variables
   - Improved error handling and validation
   - Added type safety with dedicated structs for measurements and parameters

3. **Enhanced Maintainability**
   - Separated concerns into distinct components
   - Improved code organization with clear component responsibilities
   - Added detailed documentation for all classes and methods
   - Used consistent naming conventions

4. **Better Testability**
   - Core algorithm can now be tested independently of ROS
   - Each component can be tested in isolation
   - Removed direct dependencies on ROS messages in the core algorithm

## New Architecture

```
SplineFusionCore           // Main class that orchestrates the fusion process
  |
  ├── Types                // Common data types and structures
  |
  ├── WindowManager        // Manages the sliding window of control points
  |
  ├── Optimizer            // Handles optimization of the spline
  |
  ├── DataProcessor        // Prepares data and performs IMU integration
  |
  └── SplineState          // Manages the B-spline representation
```

## ROS Integration

To maintain compatibility with ROS, we've created an adapter layer:

```
SplineFusionAdapter        // Adapts the core library to ROS
  |
  └── SplineFusionCore     // Uses the core library
```

## Benefits

1. **Easier Maintenance**: By separating the core algorithm from ROS dependencies, the code is easier to maintain and update.

2. **Improved Stability**: The core algorithm is now more robust and less prone to errors due to better encapsulation and data validation.

3. **Easier Testing**: Each component can be tested in isolation, making it easier to identify and fix issues.

4. **Framework Independence**: The core algorithm can now be used with different frameworks, not just ROS.

5. **Better Code Organization**: The code is now organized in a more logical and intuitive way, making it easier to understand and modify.

## Migration to ROS2

With this refactoring, migrating to ROS2 will be much simpler:

1. Only the adapter layer needs to be modified, not the core algorithm.
2. The well-defined interfaces make it clear what needs to be changed.
3. The separation of concerns makes it easier to adapt to the new message types and lifecycle management in ROS2.

## Future Improvements

1. **Complete the Removal of ROS Dependencies**: Some temporary functions that still use ROS types could be fully removed.
2. **Optimize Performance**: Further performance improvements could be made, especially in the optimization process.
3. **Add Unit Tests**: Comprehensive unit tests for each component would improve reliability.
4. **Improve Documentation**: More detailed documentation would make the codebase even easier to understand and use.
