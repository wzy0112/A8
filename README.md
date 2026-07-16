Simulation Baseline file is scripts/Baseline_visuliazation_A8_OD/OD_reroute_calibrated, at first run generate_od_reroute_traffic_A8.py then run run_sum_gui_A8_heatmap_data_output_OD_reroute.py.
The output of Simulation Baseline is inheat_edges
Simulation Flooding is scripts/Generate_flooding_A8_OD/generate_flooding_A8_OD_reroute_nottest_calibrated, three level is flooding_full_closure_A8_nottest_OD_reroute.py, flooding_lane_closure_A8_nottest_OD_reroute.py, flooding_scanario_A8_nottest_OD_reroute.py
The output of Simulation Flooding is in heat_edges
Build network file is scripts/build_network/build_network_A8, run  download_osm_A8.py-> build_sumo_network_A8.py-> build_corridor_network_A8.py
The output of build network is in sumo
Location SENSORS are in scripts/Detector_A8_OD/BASt/add detector merged/red and blue for visualize.py
The output of sensors generation is in detectors
