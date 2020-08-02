licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "multi_hand_tracking_tflite",
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/hand_tracking:multi_hand_desktop_tflite_calculators",
    ],
)

cc_binary(
    name = "multi_hand_tracking_cpu",
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/hand_tracking:multi_hand_desktop_tflite_calculators",
    ],
)

# Linux only
cc_binary(
    name = "orchestration_gpu",
    deps = [
        "//mediapipe/projects/orchestration:orchestration_gpu_main",
        "//mediapipe/projects/orchestration/graphs:multi_hand_mobile_calculators",
    ],
)

cc_library(
    name = "orchestration_x11",
    hdrs = ["orchestration_x11.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "orchestration_gpu_main",
    srcs = ["orchestration_gpu_main.cc"],
    deps = [
        ":orchestration_x11",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:classification_cc_proto",
    ],
)
