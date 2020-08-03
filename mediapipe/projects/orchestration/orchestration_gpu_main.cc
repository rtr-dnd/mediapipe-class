// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#include <cstdlib>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputStream[] = "input_video";
// constexpr char kOutputStream[] = "output_video";
constexpr char lOutputStream[] = "multi_hand_landmarks";
constexpr char hOutputStream[] = "multi_hand_rects";
constexpr char handednessOutputStream[] = "multi_hand_handedness";
// constexpr char pOutputStream[] = "multi_palm_rects_pass";
// palm_rects shouldn't be used: https://github.com/google/mediapipe/issues/734
// constexpr char dOutputStream[] = "multi_palm_detections";
constexpr char kWindowName[] = "MediaPipe";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  /*
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));
                   */
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller2,
                   graph.AddOutputStreamPoller(handednessOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller3,
                   graph.AddOutputStreamPoller(hOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller4,
                    graph.AddOutputStreamPoller(lOutputStream));
                  /*
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller4,
                   graph.AddOutputStreamPoller(pOutputStream));
                   */
                   /*
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller5,
                   graph.AddOutputStreamPoller(dOutputStream));
                   */
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) break;  // End of video.
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return ::mediapipe::OkStatus();
        }));

    // Get the graph result packet, or stop if that fails.
    // mediapipe::Packet packet;
    mediapipe::Packet packet2;
    mediapipe::Packet packet3;
    mediapipe::Packet packet4;
    // mediapipe::Packet packet5;
    // if (!poller.Next(&packet)) break;
    if (!poller2.Next(&packet2)) break;
    if (!poller3.Next(&packet3)) break;
    if (!poller4.Next(&packet4)) break;
    // if (!poller5.Next(&packet5)) break;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;

    // ------------ PROCESS AREA! ------------
    std::vector<mediapipe::NormalizedRect> handRects = packet3.Get<std::vector<mediapipe::NormalizedRect>>();
    float rectCenterX = handRects[0].x_center();
    float rectCenterX2 = 0;
    int leftIndex = -1; // index of left hand (-1 if not detected) : -1 ~ 1
    int rightIndex = -1; // index of right hand (-1 if not detected) : -1 ~ 1
    float indexCosScore = 0; // openness of index finger: around -0.3~0.3
    float openness = 0; // openness of index finger: exactly -1 ~ 1
    static float prevIndexPseudoVelocity = 0;
    const int historyNum = 10;
    static std::vector<float> velocityHistory(historyNum, 0.0);
    static int peakCount = 0;
    static bool wasOpen = false; // open/close gesture trigger

    std::vector<mediapipe::ClassificationList> handedness;
    std::vector<mediapipe::NormalizedLandmarkList> landmarkList;
    if (rectCenterX >= 0.1) {
      handedness = packet2.Get<std::vector<mediapipe::ClassificationList>>();
      landmarkList = packet4.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
      switch(handRects.size()) {
          case 1:
              // std::cout << handRects.size() << " " << rectCenterX << "\n";
              if (handedness[0].classification(0).label() == "left ") {
                leftIndex = 0;
              } else {
                rightIndex = 0;
              }
              break;
          case 2:
              rectCenterX2 = handRects[1].x_center();
              // std::cout << handRects.size() << " " << rectCenterX << " " << rectCenterX2 << "\n";
              if (handedness[0].classification(0).label() == "left") {
                leftIndex = 0;
                rightIndex = 1;
              } else {
                leftIndex = 1;
                rightIndex = 0;
              }
              break;
      }
      if (rightIndex >= 0) {
        std::vector<float> indexTipV = {
          landmarkList[rightIndex].landmark(8).x() - landmarkList[rightIndex].landmark(7).x(),
          landmarkList[rightIndex].landmark(8).y() - landmarkList[rightIndex].landmark(7).y()
        };
        std::vector<float> middleTipV = {
          landmarkList[rightIndex].landmark(12).x() - landmarkList[rightIndex].landmark(11).x(),
          landmarkList[rightIndex].landmark(12).y() - landmarkList[rightIndex].landmark(11).y()
        };
        std::vector<float> ringTipV = {
          landmarkList[rightIndex].landmark(16).x() - landmarkList[rightIndex].landmark(15).x(),
          landmarkList[rightIndex].landmark(16).y() - landmarkList[rightIndex].landmark(15).y()
        };
        if ((
          indexTipV[0]*middleTipV[0] + indexTipV[1]*middleTipV[1] > 0 &&
          indexTipV[0]*ringTipV[0] + indexTipV[1]*ringTipV[1] > 0
        )) { // open close
          std::vector<float> indexPalmV = {
            landmarkList[rightIndex].landmark(5).x() - landmarkList[rightIndex].landmark(0).x(),
            landmarkList[rightIndex].landmark(5).y() - landmarkList[rightIndex].landmark(0).y()
          };
          std::vector<float> indexMiddleV = {
            landmarkList[rightIndex].landmark(7).x() - landmarkList[rightIndex].landmark(6).x(),
            landmarkList[rightIndex].landmark(7).y() - landmarkList[rightIndex].landmark(6).y()
          };
          indexCosScore = 
            (indexPalmV[0]*indexMiddleV[0] + indexPalmV[1]*indexMiddleV[1])
            / sqrt(std::pow(indexPalmV[0], 2) + std::pow(indexPalmV[1], 2))
            / sqrt(std::pow(indexPalmV[0], 2) + std::pow(indexPalmV[1], 2));
          openness = std::min(std::abs(indexCosScore*0.5/0.3 + 0.5), 1.0);
          std::cout << "open close \n";
          wasOpen = true;
          if(openness < 0.3) {
            std::cout << "killing all \n";
            std::system("killall play");
          }
        } else { //airtap
          std::cout << "airtap \n";
          if (prevIndexPseudoVelocity != 0) {
            for( int i = 1; i < historyNum; i++ ) {
              velocityHistory[historyNum - i] = velocityHistory[historyNum - i - 1];
            }
            velocityHistory[0] = sqrt(
              std::pow(landmarkList[rightIndex].landmark(8).y(), 2) + 
              std::pow(landmarkList[rightIndex].landmark(8).x(), 2)) / prevIndexPseudoVelocity;
            if ( peakCount <= 0 && 
              // gesture detected &&
              *max_element(velocityHistory.begin(), velocityHistory.end()) > 1.1 &&
              *min_element(velocityHistory.begin(), velocityHistory.end()) < 0.93
            ) {
              std::cout << "peak: ";
              if((landmarkList[rightIndex].landmark(8).x() - landmarkList[rightIndex].landmark(7).x()) < 0) {
                if(landmarkList[rightIndex].landmark(8).y()*input_frame_mat.cols < input_frame_mat.rows/2) {
                  std::cout << "upper left \n";
                  std::system("play -q ~/Music/1.mp3 &"); // upper left
                } else {
                  std::cout << "under left \n";
                  std::system("play -q ~/Music/2.mp3 &"); // under left
                }
              } else { 
                std::cout << "right "; 
                if(landmarkList[rightIndex].landmark(8).y()*input_frame_mat.cols < input_frame_mat.rows/2) {
                  std::cout << "upper right \n";
                  std::system("play -q ~/Music/3.mp3 &"); // upper right
                } else {
                  std::cout << "under right \n";
                  std::system("play -q ~/Music/4.mp3 &"); // under right
                }
              }
              peakCount = 10;
            } else {
              if (peakCount >= 0) {peakCount--;}
            }
            std::cout << velocityHistory[0] << "\n";
          }
          prevIndexPseudoVelocity = sqrt(std::pow(landmarkList[rightIndex].landmark(8).y(), 2)
          + std::pow(landmarkList[rightIndex].landmark(8).x(), 2));
        }
      } else {
        prevIndexPseudoVelocity = 0;
      }
      if (leftIndex >= 0) {
        // detect gesture
        // if (/*gesturedetected*/) {
          // turn up the volume
        // }
      }
    } 


    // Convert GpuBuffer to ImageFrame.
    /*
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
          auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
          auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
          output_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
              gpu_frame.width(), gpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          gpu_helper.BindFramebuffer(texture);
          const auto info =
              mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
          glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                       info.gl_type, output_frame->MutablePixelData());
          glFlush();
          texture.Release();
          return ::mediapipe::OkStatus();
        }));
        */

    // ----------- RENDER AREA! -----------
    // Convert back to opencv for display or saving.
    // cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    cv::Mat output_frame_mat = input_frame_mat.clone();
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR); // RGB from here on
    cv::Point centerPoint;
    centerPoint.x = output_frame_mat.cols/2; centerPoint.y = output_frame_mat.rows/2;
    // cv::circle(output_frame_mat, centerPoint, 50, (0, 0, 255), 3);
    // cv::circle(output_frame_mat, centerPoint, 50*std::abs(indexCosScore*180.0/3.14159265358979*1.5+90)/180, (255, 0, 0), -1);
    // cv::circle(output_frame_mat, centerPoint, 50*openness + 1, (255, 0, 0), -1);

    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(FLAGS_output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      // if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
      if (pressed_key == 27) grab_frames = false;
    }
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
