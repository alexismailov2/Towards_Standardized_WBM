project(tests)

add_executable(${PROJECT_NAME}
        WilsonSimulation_tests.cpp)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        WilsonSimulation)