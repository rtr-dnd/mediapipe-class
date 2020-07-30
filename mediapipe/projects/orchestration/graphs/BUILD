load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])  # apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files(glob([
    "*.pbtxt",
]))

cc_library(
    name = "desktop_offline_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/core:packet_inner_join_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
    ],
)

cc_library(
    name = "desktop_tflite_calculators",
    deps = [
        ":desktop_offline_calculators",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_detection_cpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_landmark_cpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:renderer_cpu",
    ],
)

cc_library(
    name = "mobile_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_detection_gpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_landmark_gpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:renderer_gpu",
        "//mediapipe/projects/pinchlens/calculators:my_pass_through_calculator",
    ],
)

mediapipe_binary_graph(
    name = "hand_tracking_mobile_gpu_binary_graph",
    graph = "hand_tracking_mobile.pbtxt",
    output_name = "hand_tracking_mobile_gpu.binarypb",
    deps = [":mobile_calculators"],
)

cc_library(
    name = "multi_hand_desktop_tflite_calculators",
    deps = [
        ":desktop_offline_calculators",
        "//mediapipe/calculators/util:association_norm_rect_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:multi_hand_detection_cpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:multi_hand_landmark_cpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:multi_hand_renderer_cpu",
    ],
)

mediapipe_simple_subgraph(
    name = "multi_hand_landmark_gpu",
    graph = "multi_hand_landmark.pbtxt",
    register_as = "MultiHandLandmarkSubgraph",
    deps = [
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_landmark_gpu",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/util:filter_collection_calculator",
    ],
)

cc_library(
    name = "multi_hand_mobile_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/util:association_norm_rect_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:multi_hand_detection_gpu",
        "//mediapipe/projects/orchestration/graphs:multi_hand_landmark_gpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:multi_hand_renderer_gpu",
        "//mediapipe/calculators/core:pass_through_calculator",
    ],
)


mediapipe_binary_graph(
    name = "multi_hand_tracking_mobile_gpu_binary_graph",
    graph = "multi_hand_tracking_mobile.pbtxt",
    output_name = "multi_hand_tracking_mobile_gpu.binarypb",
    deps = [":multi_hand_mobile_calculators"],
)

cc_library(
    name = "detection_mobile_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_detection_gpu",
        "//mediapipe/graphs/hand_tracking/subgraphs:renderer_gpu",
    ],
)

mediapipe_binary_graph(
    name = "hand_detection_mobile_gpu_binary_graph",
    graph = "hand_detection_mobile.pbtxt",
    output_name = "hand_detection_mobile_gpu.binarypb",
    deps = [":detection_mobile_calculators"],
)