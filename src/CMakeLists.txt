project(WilsonSimulation)

# Collect all public headers
file(GLOB_RECURSE ${PROJECT_NAME}_PUBLIC_HEADERS include/*.hpp)

# Collect all sources from all folders to the whole list(temporary)
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCES *.cpp *.hpp)

# Create our target, sice it is a main library let it be the same name as a project name.
add_library(${PROJECT_NAME}
        ${${PROJECT_NAME}_SOURCES})

# Public headers of library should be a public property
target_include_directories(${PROJECT_NAME}
        PUBLIC
        include)

# Another includes which will not be available from any header files should be private
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${DEPENDENCIES_PREFIX_PATH}/bayesopt/include
        #${GSL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        #${GSL_INCLUDE_DIRECTORIES}
        ${CMAKE_CURRENT_LIST_DIR})

# Private or public but in case it will be used outside of this cmake it will need all dependencies.
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        -L${DEPENDENCIES_PREFIX_PATH}/bayesopt/lib
        libbayesopt.a
        libnlopt.a)
        #${GSL_LIBRARIES})

# Set all needed public headers to be delivered to installation path
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADERS "${${PROJECT_NAME}_PUBLIC_HEADERS}")

# Install usual structure include, bin, lib
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${LIBRARY_INSTALL_DESTINATION}/lib
        LIBRARY DESTINATION ${LIBRARY_INSTALL_DESTINATION}/lib
        RUNTIME DESTINATION ${LIBRARY_INSTALL_DESTINATION}/bin
        PUBLIC_HEADER DESTINATION ${LIBRARY_INSTALL_DESTINATION}/include/${PROJECT_NAME})