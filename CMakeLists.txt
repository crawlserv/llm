# cMake version
cmake_minimum_required(VERSION 3.8.2)

# project name
set(PROJECT_NAME llm)
project(${PROJECT_NAME})

# show source directory
message("Source: \"${CMAKE_CURRENT_SOURCE_DIR}/\"")

# find Multi-threading library
find_package(Threads REQUIRED)

message(${Threads_FOUND})

if(${Threads_FOUND} AND NOT CMAKE_THREAD_LIBS_INIT STREQUAL "")
	message("Multi-threading library found: \"${CMAKE_THREAD_LIBS_INIT}\"")
elseif(${Threads_FOUND})
	message("Multi-threading library included in system libraries.")
else()
	message(WARNING "Multi-threading library not found.")
endif()

# find libcurl
find_package(curl QUIET)

if(${curl_FOUND} AND curl_LIBRARIES)
	message("libcurl ${curl_VERSION} library found: \"${curl_LIBRARIES}\"")
else()
	# find manually
	unset(curl_LIBRARIES)

	find_library(curl_LIBRARIES curl PATH_SUFFIXES ${CMAKE_LIBRARY_ARCHITECTURE} lib64 lib REQUIRED)

	if(curl_LIBRARIES)
		message("libcurl manually found: \"${curl_LIBRARIES}\"")
	else()
		message(WARNING "libcurl not found.")
	endif()
endif()

# set include directory for jsoncons to sub-directory of source code
set(jsoncons_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/src/_extern/jsoncons/include")

message("jsoncons include directory set to: \"${jsoncons_INCLUDE}\".")

# source files
add_executable(${PROJECT_NAME}
		src/main.cpp
		src/Wrapper/Curl.cpp
)

# C++ 17 (including corresponding standard library) is required
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# set include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${jsoncons_INCLUDE})

# set libraries
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${curl_LIBRARIES})

# set required library options
target_compile_definitions(${PROJECT_NAME} PRIVATE RAPIDJSON_HAS_STDSTRING)
target_compile_definitions(${PROJECT_NAME} PRIVATE RAPIDJSON_NO_SIZETYPEDEFINE)

# set compiler options
if(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /W4)

	if(${CMAKE_BUILD_TYPE} AND ${CMAKE_BUILD_TYPE} STREQUAL Debug)
		message("Using MSVC without optimiziation and with debugging information.")
		target_compile_options(${PROJECT_NAME} PRIVATE /Od)
		target_compile_options(${PROJECT_NAME} PRIVATE /Zi)
	else()
		message("Using MSVC with maximum optimiziation and without debugging information.")
		target_compile_options(${PROJECT_NAME} PRIVATE /Ox)
	endif()
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

    if(NOT CMAKE_BUILD_TYPE)
    	SET(CMAKE_BUILT_TYPE Release ... FORCE)
    endif()

	if(${CMAKE_BUILD_TYPE} MATCHES Debug)
		message("Using \"${CMAKE_CXX_COMPILER}\" with minimum optimization and debugging information.")
		target_compile_options(${PROJECT_NAME} PRIVATE -g3)
		target_compile_options(${PROJECT_NAME} PRIVATE -O0)
	else()
		message("Using \"${CMAKE_CXX_COMPILER}\" with maximum optimization and without debugging information.")
		target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
		target_compile_options(${PROJECT_NAME} PRIVATE -g0)
		target_compile_options(${PROJECT_NAME} PRIVATE -O3)
	endif()
endif()

# set optional library options
target_compile_definitions(${PROJECT_NAME} PRIVATE JSONCONS_NO_DEPRECATED)

# set Eclipse-specific temporary compiler option (see https://bugs.eclipse.org/bugs/show_bug.cgi?id=500798)
if(CMAKE_ECLIPSE_VERSION)
	target_compile_options(${PROJECT_NAME} PRIVATE -ftemplate-backtrace-limit=0)
endif()

# setup clang-tidy
#find_program(CLANG_TIDY_EXE NAMES "clang-tidy" DOC "Path to clang-tidy executable" )

#if(CLANG_TIDY_EXE)
#	message("clang-tidy found: ${CLANG_TIDY_EXE}")
	
#	string(
#			CONCAT CLANG_TIDY_TESTS "*,"
#			"-altera-*,"
#			"-clang-analyzer-core.NullDereference,"
#			"-clang-analyzer-optin.cplusplus.UninitializedObject,"
#			"-clang-diagnostic-c++17-extensions,"
#			"-cppcoreguidelines-non-private-member-variables-in-classes,"
#			"-cppcoreguidelines-pro-type-reinterpret-cast,"
#			"-cppcoreguidelines-pro-type-union-access,"
#			"-fuchsia-default-arguments-calls,"
#			"-fuchsia-multiple-inheritance,"
#			"-fuchsia-overloaded-operator,"
#			"-google-readability-todo,"	
#			"-google-runtime-references,"
#			"-llvm-header-guard,"
#			"-llvmlibc-*,"
#			"-misc-non-private-member-variables-in-classes,"
#			"-modernize-pass-by-value,"
#			"-modernize-use-trailing-return-type,"
#			"-readability-identifier-length,"
#	)
#	
#	set(
#			CLANG_TIDY_HEADER_FILTER [=[[A-Z][a-zA-Z]+\.hpp$]=]
#	)
#	
#	set(
#			DO_CLANG_TIDY
#			"${CLANG_TIDY_EXE}"
#			"-header-filter=${CLANG_TIDY_HEADER_FILTER}"
#			"-checks=${CLANG_TIDY_TESTS}"
#	)
#	
#	set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}" )
#else()
#	message(WARNING "clang-tidy not found.")
#endif()

message("Run \"make\" from build directory to build.")
