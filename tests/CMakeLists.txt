find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

add_executable(unit_tests
    unit/test_pricing_engine.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        OptionsPricingLib
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        Threads::Threads
)

target_include_directories(unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME unit_tests COMMAND unit_tests)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(unit_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-g -O0 --coverage>
        $<$<CXX_COMPILER_ID:Clang>:-g -O0 --coverage>
    )
    
    target_link_options(unit_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:--coverage>
        $<$<CXX_COMPILER_ID:Clang>:--coverage>
    )
endif()

set_target_properties(unit_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)