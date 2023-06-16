cmake_minimum_required(VERSION 3.26)
project(pyWilsonSimulation)

set(CMAKE_CXX_STANDARD 14)

include(ExternalProject)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DEPENDENCIES_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/dependencies")

ExternalProject_Add(bayesopt_external
        PREFIX ${DEPENDENCIES_PREFIX_PATH}/bayesopt-tmp
        GIT_REPOSITORY "https://github.com/rmcantin/bayesopt.git"
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${DEPENDENCIES_PREFIX_PATH}/bayesopt
        -DCMAKE_BUILD_TYPE=Release
        -DHAVE_GETTIMEOFDAY=OFF
        -DBAYESOPT_BUILD_EXAMPLES=OFF
        -DCMAKE_CXX_FLAGS="-std=c++11")
set(BAYESOPT_ROOT ${DEPENDENCIES_PREFIX_PATH}/bayesopt)
list(APPEND DEPENDENCIES_LIST bayesopt_external)

# TODO: Does not work on MacOS I do not know what is the problem
#ExternalProject_Add(gsl
#        PREFIX ${DEPENDENCIES_PREFIX_PATH}/gsl-tmp
#        GIT_REPOSITORY "https://github.com/LuaDist/gsl.git")
#        #CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${DEPENDENCIES_PREFIX_PATH}/gsl -DCMAKE_INCLUDE_DIRECTORIES=${DEPENDENCIES_PREFIX_PATH}/gsl-tmp/src -DCMAKE_BUILD_TYPE=Release)
#set(GSL_ROOT ${DEPENDENCIES_PREFIX_PATH}/gsl)
#list(APPEND DEPENDENCIES_LIST gsl_external)

#find_package(gsl REQUIRED)
find_package(Boost REQUIRED)

if (NOT EXISTS "${DEPENDENCIES_PREFIX_PATH}/bayesopt/lib/libbayesopt.a")
    return()
endif()

add_subdirectory(src)

#file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/WilsonSimulation/include/WilsonSimulation
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/interface/WilsonSimulation SYMBOLIC)

set(CMAKE_SWIG_FLAGS)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

if(${SWIG_VERSION} VERSION_GREATER_EQUAL 4)
    list(APPEND CMAKE_SWIG_FLAGS "-doxygen")
endif()

if(UNIX AND NOT APPLE)
    list(APPEND CMAKE_SWIG_FLAGS "-DSWIGWORDSIZE64")
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development) #NumPy)
if(Python_VERSION VERSION_GREATER_EQUAL 3)
    list(APPEND CMAKE_SWIG_FLAGS "-py3;-DPY3")
endif()

#set(PYTHON_PROJECT cmakeswig)

set_property(SOURCE src/interface/WilsonSimulation.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE src/interface/WilsonSimulation.i PROPERTY SWIG_MODULE_NAME ${PROJECT_NAME})
swig_add_library(${PROJECT_NAME}
        TYPE SHARED
        LANGUAGE python
        SOURCES src/interface/WilsonSimulation.i
        OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR}/python/pyWilsonSimulation)

swig_link_libraries(${PROJECT_NAME}
        WilsonSimulation
        ${Python_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX .so)
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/python/${PROJECT_NAME})

# note: macOS is APPLE and also UNIX !
#if(APPLE)
#    set_target_properties(${PROJECT_NAME} PROPERTIES
#            SUFFIX ".so"
#            INSTALL_RPATH "@loader_path;@loader_path/../../${PYTHON_PROJECT}/.libs")
#    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
#            LINK_FLAGS "-flat_namespace -undefined suppress")
#elseif(UNIX)
#    set_target_properties(${PROJECT_NAME} PROPERTIES
#            INSTALL_RPATH "$ORIGIN:$ORIGIN/../../${PYTHON_PROJECT}/.libs")
#endif()

target_include_directories(${PROJECT_NAME} PRIVATE
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        WilsonSimulation/include)

# Files to install with Python
#set(PYTHON_INSTALL_FILES
#        ${CMAKE_CURRENT_BINARY_DIR}/pyWilsonSimulation.py
#        ${CMAKE_CURRENT_BINARY_DIR}/_pyWilsonSimulation.so)

# Configure setup.py and copy to output directory
configure_file(src/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_target(${PROJECT_NAME}_copy ALL
        COMMAND rm -rf ${CMAKE_CURRENT_LIST_DIR}/python/${PROJECT_NAME}/*.h ${CMAKE_CURRENT_LIST_DIR}/python/${PROJECT_NAME}/*.cxx
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Copying pyWilsonSimulation"
        DEPENDS ${PROJECT_NAME})

add_dependencies(${PROJECT_NAME}_copy ${PROJECT_NAME})

# Build target to call setup.py
add_custom_target(install-python
        DEPENDS ${PROJECT_NAME}_copy
        COMMAND python ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel)

add_dependencies(${PROJECT_NAME} ${DEPENDENCIES_LIST})

add_subdirectory(tests)