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
#include <google/protobuf/util/json_util.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>

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

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";     // MediaPipeから出力された映像
constexpr char lOutputStream[] = "output_landmarks"; // MediaPipeから出力された手の特徴量情報
constexpr char rOutputStream[] = "output_hand_rect"; // MediaPipeから出力された、映像内での手の位置
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

// アルファ値を保持したまま画像を別の画像の上に描画する
void overlayImage(cv::Mat *src, cv::Mat *overlay, const cv::Point &location)
{
  for (int y = std::max(location.y, 0); y < src->rows; ++y)
  {
    int fY = y - location.y;
    if (fY >= overlay->rows)
      break;
    for (int x = std::max(location.x, 0); x < src->cols; ++x)
    {
      int fX = x - location.x;
      if (fX >= overlay->cols)
        break;
      double opacity = ((double)overlay->data[fY * overlay->step + fX * overlay->channels() + 3]) / 255;
      for (int c = 0; opacity > 0 && c < src->channels(); ++c)
      {
        unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
        unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
        src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
      }
    }
  }
}

::mediapipe::Status RunMPPGraph()
{
  const bool pass_through = false; // パススルーモードのオンオフ

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
  if (load_video)
  {
    capture.open(FLAGS_input_video_path);
  }
  else
  {
    if (pass_through)
    {
      capture.open(1); // ウェブカム
    }
    else
    {
      capture.open(0); // 内蔵カメラ
    }
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video)
  {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  // それぞれのストリームのポーラを設定
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller2,
                   graph.AddOutputStreamPoller(lOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller3,
                   graph.AddOutputStreamPoller(rOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  // whether the hand is detected
  bool isDetected = false;
  // 今ピンチ中か否か
  bool isActive = false;
  cv::Scalar colorOfPoint = cv::Scalar(255, 255, 255, 1);
  std::string dispMessage = "";
  float distanceBetween = 100.0;
  int counter = 0;
  std::vector<cv::Point> centerPoints = {
      cv::Point(100, 100),
      cv::Point(100, 200),
  };
  float pinchThresholdLower = 0.1;
  float pinchThresholdUpper = 0.4;
  float distanceThreshold = 50;

  std::ofstream outputfile("/home/mech-user/dev/log.txt"); // File path
  auto startTime = std::chrono::system_clock::now();
  std::time_t startTimeConv = std::chrono::system_clock::to_time_t(startTime);
  outputfile << std::ctime(&startTimeConv);
  outputfile << "\n\n";

  // 直近aveNum個の距離と中点の値を保存しておき、平均を取って描画する
  // const int aveNum = 5;
  // float recentDistance[aveNum] = {};
  // float recentBetweenVX[aveNum] = {};
  // float recentBetweenVY[aveNum] = {};

  // ピンチを検出できないと増える。これが一定以下なら、さっきまでピンチしていたということなので、次のフレームで指の距離が離れていても、ピンチアウト途中だと判定
  // int failCount = 0;

  while (grab_frames)
  {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty())
      break; // End of video.
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!pass_through)
    {
      // ミラーモードなら映像を反転
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
    // MediaPipeのパケットを取得
    mediapipe::Packet packet;
    mediapipe::Packet packet2;
    mediapipe::Packet packet3;
    if (!poller.Next(&packet))
      break;
    if (!poller2.Next(&packet2))
      break;
    if (!poller3.Next(&packet3))
      break;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;

    // パケットから各特徴量を取得
    mediapipe::NormalizedLandmarkList landmarkList = packet2.Get<mediapipe::NormalizedLandmarkList>();
    mediapipe::NormalizedLandmark thumbTip = landmarkList.landmark(4);
    mediapipe::NormalizedLandmark indexTip = landmarkList.landmark(8);

    // // generating vectors for hand posture detection
    // const std::vector<float> indexBottomV = { // 人差し指付け根のベクトル
    //   landmarkList.landmark(6).x() - landmarkList.landmark(5).x(),
    //   landmarkList.landmark(6).y() - landmarkList.landmark(5).y()};
    // const std::vector<float> middleTipV = { // 中指先端のベクトル、以下同様
    //   landmarkList.landmark(12).x() - landmarkList.landmark(11).x(),
    //   landmarkList.landmark(12).y() - landmarkList.landmark(11).y()};
    // const std::vector<float> ringTipV = {
    //   landmarkList.landmark(16).x() - landmarkList.landmark(15).x(),
    //   landmarkList.landmark(16).y() - landmarkList.landmark(15).y()};
    // const std::vector<float> littleTipV = {
    //   landmarkList.landmark(20).x() - landmarkList.landmark(19).x(),
    //   landmarkList.landmark(20).y() - landmarkList.landmark(19).y()};

    // const std::vector<float> thumbDirV = { // 人差し指全体のベクトル
    //   landmarkList.landmark(4).x() - landmarkList.landmark(3).x(),
    //   landmarkList.landmark(4).y() - landmarkList.landmark(3).y()};
    // const std::vector<float> indexDirV = {
    //   landmarkList.landmark(8).x() - landmarkList.landmark(7).x(),
    //   landmarkList.landmark(8).y() - landmarkList.landmark(7).y()};
    // std::vector<float> betweenDirV = {0, 0}; // 親指と人差し指の中点の向きのベクトル
    // if(pass_through) { // パススルーなら中点そのまま
    //   betweenDirV[0] = (thumbDirV[0] + indexDirV[0])/2;
    //   betweenDirV[1] = (thumbDirV[1] + indexDirV[1])/2;
    // } else { // ミラーなら重み付き平均
    //   betweenDirV[0] = (thumbDirV[0]*9 + indexDirV[0])/10;
    //   betweenDirV[1] = (thumbDirV[1]*9 + indexDirV[1])/10;
    // }
    // const float betweenDirNorm = std::sqrt(std::pow(betweenDirV[0], 2) + std::pow(betweenDirV[1], 2)); // 中点ベクトルのノルム
    // const std::vector<float> normalizedBetweenDirV = { // 正規化された中点ベクトル
    //   float(betweenDirV[0]/betweenDirNorm),
    //   float(betweenDirV[1]/betweenDirNorm)};
    // printf("%f, %f, %f\n", normalizedBetweenDirV[0], normalizedBetweenDirV[1], std::pow(normalizedBetweenDirV[0], 2) + std::pow(normalizedBetweenDirV[1], 2)); // デバッグ用に中点位置を出力

    mediapipe::NormalizedRect handRect = packet3.Get<mediapipe::NormalizedRect>();

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
          auto &gpu_frame = packet.Get<mediapipe::GpuBuffer>();
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

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    cv::cvtColor(input_frame_mat, input_frame_mat, cv::COLOR_RGB2BGR);

    // disabling annotation by deepcopying input frame. replace with output_frame_mat to re-enable
    cv::Mat input_frame_mat_copy = input_frame_mat.clone();

    // ウィンドウ内での手の基準点の位置
    float rectCenterX = handRect.x_center() * output_frame_mat.cols;
    float rectCenterY = handRect.y_center() * output_frame_mat.rows;
    // 正規化されたピンチの位置（これは普通に中点）
    // float pinchCenterNormX = (thumbTip.x()*1 + indexTip.x())/2;
    // float pinchCenterNormY = (thumbTip.y()*1 + indexTip.y())/2;

    /*
    pinchCenterNormとbetweenDirVの違い
    pinchCenterNorm: どこを拡大するか。ちょうど中点。
    betweenDirV: 拡大したものをどこに描画するか。重み付き平均。
    */

    // float pinchCenterX = (pinchCenterNormX) * output_frame_mat.cols;
    // float pinchCenterY = (pinchCenterNormY) * output_frame_mat.rows;
    // 握っていたらisPinching
    // bool isPinching =
    //   std::inner_product(indexBottomV.begin(), indexBottomV.end(), middleTipV.begin(), 0.0f) < 0 &&
    //   std::inner_product(indexBottomV.begin(), indexBottomV.end(), ringTipV.begin(), 0.0f) < 0 &&
    //   std::inner_product(indexBottomV.begin(), indexBottomV.end(), littleTipV.begin(), 0.0f) < 0;

    // if(pinchCenterX != 0) {
    if (thumbTip.x() != 0 && indexTip.x() != 0)
    {
      isDetected = true;
      int radius = 3;
      cv::Point center;
      // 親指と人差し指の距離。手の大きさで正規化することで手の大小にかかわらずピンチ状態を判定できる
      distanceBetween = (std::pow(thumbTip.x() - indexTip.x(), 2) + std::pow(thumbTip.y() - indexTip.y(), 2)) / handRect.width() / handRect.width();
      std::vector<float> thumbCoordinate = {
          thumbTip.x() * output_frame_mat.cols,
          thumbTip.y() * output_frame_mat.rows};
      std::vector<float> indexCoordinate = {
          indexTip.x() * output_frame_mat.cols,
          indexTip.y() * output_frame_mat.rows};
      float distanceToPointThumb = std::sqrt(std::pow(thumbCoordinate[0] - centerPoints[counter].x, 2) + std::pow(thumbCoordinate[1] - centerPoints[counter].y, 2));
      float distanceToPointIndex = std::sqrt(std::pow(indexCoordinate[0] - centerPoints[counter].x, 2) + std::pow(indexCoordinate[1] - centerPoints[counter].y, 2));
      // if (isPinching) {
      //   center.x = pinchCenterX;
      //   center.y = pinchCenterY;

      //   // 中点、距離についてaveNum個の平均をとる
      //   float distanceAverage = 0;
      //   float betweenVAverageX = 0;
      //   float betweenVAverageY = 0;
      //   for (int i = 1; i<aveNum; i++) {
      //     float temp = recentDistance[aveNum - i - 1];
      //     recentDistance[aveNum - i] = temp;
      //     distanceAverage += temp;
      //     temp = recentBetweenVX[aveNum - i - 1];
      //     recentBetweenVX[aveNum - i] = temp;
      //     betweenVAverageX += temp;
      //     temp = recentBetweenVY[aveNum - i - 1];
      //     recentBetweenVY[aveNum - i] = temp;
      //     betweenVAverageY += temp;
      //   }
      //   recentDistance[0] = distanceBetween;
      //   distanceAverage += distanceBetween;
      //   distanceAverage /= aveNum;
      //   recentBetweenVX[0] = normalizedBetweenDirV[0];
      //   betweenVAverageX += normalizedBetweenDirV[0];
      //   betweenVAverageX /= aveNum;
      //   recentBetweenVY[0] = normalizedBetweenDirV[1];
      //   betweenVAverageY += normalizedBetweenDirV[1];
      //   betweenVAverageY /= aveNum;

      if (isActive)
      {
        if (distanceBetween >= pinchThresholdUpper)
        {
          isActive = false;
          counter++;
          outputfile << "-\n";
          colorOfPoint = cv::Scalar(0, 255, 0, 1);
          dispMessage = "Hand detected";
        }
        else if ((distanceToPointThumb <= distanceThreshold) && (distanceToPointIndex <= distanceThreshold) && (distanceBetween <= pinchThresholdLower))
        {
          isActive = true;
          colorOfPoint = cv::Scalar(255, 0, 0, 1);
          dispMessage = "Pinch";
        }
      }
      else if ((distanceToPointThumb <= distanceThreshold) && (distanceToPointIndex <= distanceThreshold) && (distanceBetween <= pinchThresholdLower))
      {
        isActive = true;
        colorOfPoint = cv::Scalar(255, 0, 0, 1);
        dispMessage = "Pinch";
      }
      else
      {
        colorOfPoint = cv::Scalar(0, 255, 0, 1);
        dispMessage = "Hand detected";
      }

      if (isActive)
      {
        outputfile << distanceToPointThumb;
        outputfile << ", ";
        outputfile << distanceToPointIndex;
        outputfile << ", ";
        outputfile << distanceBetween;
        outputfile << "\n";
      }

      // nothing touched from here
      // if (failCount <= 2) {
      //   // draw pinch window
      //   int ROISize = std::min(int(distanceAverage * 1000), 400);
      //   // distanceAverage: 0 ~ 0.8 (approx)
      //   float lensScale = distanceAverage/0.8*10 + 1;
      //   cv::resize(input_frame_mat, input_frame_mat, cv::Size(), lensScale, lensScale);
      //   cv::Rect ROI(cv::Point(lensScale*center.x - ROISize/2, lensScale*center.y - ROISize/2), cv::Size(ROISize, ROISize));

      //   // クロップされた拡大映像が入る
      //   cv::Mat cropped = input_frame_mat(ROI);

      //   cv::Mat mask(cropped.size().width, cropped.size().height, CV_8UC1);//circle
      //   cv::rectangle(mask, cv::Point(0, 0), cv::Point(cropped.size().width, cropped.size().height), cv::Scalar(0), -1);
      //   cv::circle(mask, cv::Point(cropped.size().width/2, cropped.size().height/2), cropped.size().width/2, cv::Scalar(255), -1, 8, 0);
      //   cv::Mat masked(cropped.size().width, cropped.size().height, CV_8UC4);//dist
      //   cv::Mat srcArray[] = {cropped, mask};
      //   int from_to[] = {0,0, 1,1, 2,2, 3,3};
      //   cv::mixChannels(srcArray, 2, &masked, 1, from_to, 4);
      //   // ここまでで周囲が透明な円形ウィンドウができる

      //   // 中点からずらされた描画位置
      //   cv::Point centerOffset;
      //   centerOffset.x = center.x - int(ROISize/2) + betweenVAverageX*int(ROISize/1.5);
      //   centerOffset.y = center.y - int(ROISize/2) + betweenVAverageY*int(ROISize/1.5);
      //   overlayImage(&input_frame_mat_copy, &masked, centerOffset); // 円形ウィンドウを描画
      //   centerOffset.x = center.x + betweenVAverageX*int(ROISize/1.5);
      //   centerOffset.y = center.y + betweenVAverageY*int(ROISize/1.5);
      //   cv::circle(input_frame_mat_copy, centerOffset, cropped.size().width/2, cv::Scalar(255, 255, 255, 0.5), distanceAverage/0.8*3, cv::LINE_AA, 0); // 白色の枠線を描画
      //   cv::circle(input_frame_mat_copy, cv::Point(15, 22), 5, cv::Scalar(222, 137, 18), -1, cv::LINE_AA, 0); // 左上の青いインジケーター
      //   cv::putText(input_frame_mat_copy, "Pinching", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200,200,200), 1, cv::LINE_AA); //
      // }
      // } else { // ピンチしていないとき
      //   isActive = false;
      //   // failCount += 1;
      //   cv::circle(input_frame_mat_copy, cv::Point(15, 22), 5, cv::Scalar(78, 184, 17), -1, cv::LINE_AA, 0); // 左上の緑のインジケーター
      //   cv::putText(input_frame_mat_copy, "Hand detected", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200,200,200), 1, cv::LINE_AA); // 左上のテキスト
      // }
    }
    else
    { // 手がないとき
      isActive = false;
      // failCount += 1;
      colorOfPoint = cv::Scalar(0, 0, 255, 1);
      dispMessage = "Hand not detected";
    }

    cv::circle(input_frame_mat_copy, centerPoints[counter], 5, colorOfPoint, -1, cv::LINE_AA, 0);
    cv::putText(input_frame_mat_copy, dispMessage, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1, cv::LINE_AA); //
    if (isDetected && isActive)
    {
      cv::putText(input_frame_mat_copy, std::to_string(distanceBetween), centerPoints[counter], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1, cv::LINE_AA); //
    }

    if (save_video)
    {
      if (!writer.isOpened())
      {
        LOG(INFO) << "Prepare video writer.";
        writer.open(FLAGS_output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
                    capture.get(cv::CAP_PROP_FPS), input_frame_mat_copy.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(input_frame_mat_copy);
    }
    else
    {
      cv::imshow(kWindowName, input_frame_mat_copy);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255)
        grab_frames = false;
    }
  }

  outputfile.close();

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened())
    writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok())
  {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  }
  else
  {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}