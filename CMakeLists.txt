cmake_minimum_required(VERSION 3.20)
project(OptionsOricingEngine 
    VERSION 1.0.0
    DESCRIPTION "High-Performance Low-Latency Options Pricing Engine"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(BUILD_EXAMPLES "Build example applications" ON)
option(ENABLE_VECTORIZATION "Enable SIMD vectorization" ON)
option(ENABLE_LTO "Enable Link Time Optimization" ON)
option(ENABLE_NATIVE_ARCH "Optimize for native architecture" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
    if(ENABLE_LTO)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

if(ENABLE_NATIVE_ARCH)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-march=native -mtune=native)
    endif()
endif()

if(ENABLE_VECTORIZATION)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-mavx2 -mfma)
        add_compile_definitions(ENABLE_AVX2)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/arch:AVX2)
        add_compile_definitions(ENABLE_AVX2)
    endif()
endif()

add_compile_options(
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
)

find_package(Threads REQUIRED)

include_directories(include)

set(HEADERS
    include/types/Option.hpp
    include/types/Market.hpp
    include/types/Results.hpp
    include/math/NormalDistribution.hpp
    include/math/Statistics.hpp
    include/math/Optimization.hpp
    include/options/BlackScholes.hpp
    include/options/Greeks.hpp
    include/options/MonteCarlo.hpp
    include/options/AmericanOptions.hpp
    include/options/ImpliedVolatility.hpp
    include/options/PricingEngine.hpp
    include/utils/Timer.hpp
    include/utils/MemoryPool.hpp
    include/utils/ThreadPool.hpp
)

set(SOURCES
    src/main.cpp
    src/math/Statistics.cpp
    src/math/NormalDistribution.cpp
    src/options/BlackScholes.cpp
    src/options/Greeks.cpp
    src/options/PricingEngine.cpp
    src/utils/Timer.cpp
)

add_library(OptionsPricingLib STATIC ${SOURCES} ${HEADERS})

target_include_directories(OptionsPricingLib 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(OptionsPricingLib 
    PUBLIC 
        Threads::Threads
)

target_compile_features(OptionsPricingLib PUBLIC cxx_std_20)

add_executable(OptionsOricingEngine src/main.cpp)
target_link_libraries(OptionsOricingEngine PRIVATE OptionsPricingLib)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest)
    
    if(GTest_FOUND)
        add_subdirectory(tests)
    else()
        message(WARNING "Google Test not found. Tests will not be built.")
    endif()
endif()

if(BUILD_BENCHMARKS)
    find_package(benchmark)
    
    if(benchmark_FOUND)
        add_subdirectory(benchmarks)
    else()
        message(WARNING "Google Benchmark not found. Benchmarks will not be built.")
    endif()
endif()

install(TARGETS OptionsPricingLib OptionsOricingEngine
    EXPORT OptionsPricingTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ 
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT OptionsPricingTargets
    FILE OptionsPricingTargets.cmake
    NAMESPACE OptionsПрicing::
    DESTINATION lib/cmake/OptionsПricing
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/OptionsPricingConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OptionsPricingConfig.cmake
    INSTALL_DESTINATION lib/cmake/OptionsПricing
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/OptionsPricingConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OptionsPricingConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OptionsPricingConfigVersion.cmake
    DESTINATION lib/cmake/OptionsПricing
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_target(package
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install
        COMMAND ${CMAKE_COMMAND} -E tar czf options-pricing-engine-${PROJECT_VERSION}.tar.gz
            ${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Creating package"
    )
endif()

set(CPACK_PACKAGE_NAME "OptionsПricingEngine")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "Quantitative Developer")
set(CPACK_GENERATOR "TGZ;ZIP")

include(CPack)