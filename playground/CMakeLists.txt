target_add_executable(ifs_visualizer)
target_link_libraries(ifs_visualizer PRIVATE
    IFSLib
    imgui::imgui
)
target_sources(ifs_visualizer PRIVATE ifs_visualizer_main.cpp)

# Modular IFS visualizer
target_add_executable(ifs_modular)
target_link_libraries(ifs_modular PRIVATE
    IFSLib
    imgui::imgui
)
target_sources(ifs_modular PRIVATE ifs_modular_main.cpp)
