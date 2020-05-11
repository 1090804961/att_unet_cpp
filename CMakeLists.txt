cmake_minimum_required(VERSION 3.15)
project(att_unet_cpp)

set(CMAKE_CXX_STANDARD 14)
#CMAKE_PREFIX_PATH 可能和设置libtorch的路径相关

#不知道有没有用，如果torch无法include就试试
#include_directories(SYSTEM "C:/third/cuda10.1/libtorch-win-shared-with-deps-1.4.0/libtorch/lib")


#"C:\third\cuda10.2\libtorch\share\cmake\Torch"
set(Torch_DIR "C:/third/cuda10.1/libtorch-win-shared-with-deps-1.4.0/libtorch/share/cmake/Torch")#libtorch cuda目录
#set(Torch_DIR C:/third/cpu10.1/libtorch-win-shared-with-deps-1.4.0none/libtorch/share/cmake/Torch)#libtorch cpu目录
set(OpenCV_DIR C:/Program Files/OpenCV/CMake_OpenCV/build/install/x64/vc16/lib) #opencv目录

find_package(Torch REQUIRED) #libtorch 寻找库
find_package(OpenCV REQUIRED) #opencv 寻找库


add_executable(att_unet_cpp 01_sample_loadbyopecv.cpp)

target_link_libraries(att_unet_cpp ${OpenCV_LIBS})
target_link_libraries(att_unet_cpp "${TORCH_LIBRARIES}")#libtorch 关联库

#INCLUDE_DIRECTORIES(/usr/local/include ${PROJET_SOURCE_DIR}/Utils) #用于包含某个目录 的文件夹 没有用到

#set_property(TARGET main.cpp PROPERTY CMAKE_CXX_STANDARD 14) #设置某个工程文件的 c++规范，没啥用
#set_property(TARGET att_unet_cpp PROPERTY TORCH_LIB_DIR "C:/third/cuda10.1/libtorch-win-shared-with-deps-1.4.0/libtorch/lib")

if(MSVC)
    #    C:/Program Files/OpenCV/opencv/build/x64/vc15/bin/*.dll
    #C:/Program Files/OpenCV/CMake_OpenCV/build/install/x64/vc16/bin
    file(GLOB OPENCV_DLLS "C:/Program Files/OpenCV/CMake_OpenCV/build/install/x64/vc16/bin/*.dll")
    file(GLOB TROCH_DLLS "C:/third/cuda10.1/libtorch-win-shared-with-deps-1.4.0/libtorch/lib/*.dll") #cuda
#    file(GLOB TROCH_DLLS "C:/third/cpu10.1/libtorch-win-shared-with-deps-1.4.0none/libtorch/lib/*.dll") #cpu
    add_custom_command(
            TARGET att_unet_cpp
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            ${TROCH_DLLS}
            $<TARGET_FILE_DIR:att_unet_cpp>
    )
endif(MSVC)


