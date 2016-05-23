add_executable(nd main.cpp)
target_link_libraries( nd ${OpenCV_LIBS} )

add_executable(set1 set1.cpp)
target_link_libraries( set1 ${OpenCV_LIBS} )

add_executable(set3 set3.cpp)
target_link_libraries( set3 ${OpenCV_LIBS} )
