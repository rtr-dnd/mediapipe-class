load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])


cc_library(
    name = "end_loop_calculator",
    srcs = ["end_loop_calculator.cc"],
    hdrs = ["end_loop_calculator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:render_data_cc_proto",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)