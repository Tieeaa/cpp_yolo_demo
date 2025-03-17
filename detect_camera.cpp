#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "yolov8.h"

using namespace std;
using namespace cv;

void open_camera() {
  avdevice_register_all();
  AVFormatContext* fmt_ctx{};
  const AVInputFormat* input_fmt{};
  AVDictionary* options{};
  AVStream* video_stream{};
  int video_stream_idx{-1};
  const AVCodec* video_decoder{};
  AVCodecContext* video_decoder_ctx{};
  AVFrame* frame{};
  AVPacket* packet{};
  atomic_bool quit{false};
  SwsContext* sws_ctx{};
  int frame_height{480};
  int frame_width{640};
  Mat img(frame_height, frame_width, CV_8UC3, Scalar ::all(0));
  uint8_t* data[4]{img.data};
  int linesize[4]{static_cast<int>(img.step)};
  int frame_idx{0};

  YOLOV8 yolo{};

  // 1.open camera
  av_dict_set(&options, "video_size", "640x480", 0);
  av_dict_set(&options, "frame_rate", "30", 0);
#ifdef _WIN32
  input_fmt = av_find_input_format("dshow");
  // !!! 将2K Camera改成摄像头名称
  if (avformat_open_input(&fmt_ctx, "video=2K Camera", input_fmt, &options) <
      0) {
    cerr << "failed to open 2K Camera" << endl;
    goto end;
  }
#else
  input_fmt = av_find_input_format("x416");
  if (avformat_open_input(&fmt_ctx, "/dev/video0", input_fmt, &options) < 0) {
    cerr << "failed to open /dev/video0" << endl;
    goto end;
  }
#endif
  av_dict_free(&options);
  // 2.find video stream

  if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
    cerr << "failed to find stream info" << endl;
    goto end;
  }

  for (auto idx = 0; idx < fmt_ctx->nb_streams; idx++) {
    if (AVMEDIA_TYPE_VIDEO == fmt_ctx->streams[idx]->codecpar->codec_type) {
      video_stream_idx = idx;
      video_stream = fmt_ctx->streams[idx];
      break;
    }
  }

  if (video_stream == nullptr) {
    cerr << "failed to find video stream" << endl;
    goto end;
  }

  // 3.find decoder
  video_decoder = avcodec_find_decoder(video_stream->codecpar->codec_id);
  if (video_decoder == nullptr) {
    cerr << "failed to find video decoder" << endl;
    goto end;
  }
  // 4.configure decodeer
  video_decoder_ctx = avcodec_alloc_context3(video_decoder);
  if (video_decoder_ctx == nullptr) {
    cerr << "failed to allocate video decoder ctx" << endl;
    goto end;
  }
  if (avcodec_parameters_to_context(video_decoder_ctx, video_stream->codecpar) <
      0) {
    cerr << "failed to set decode parameters for video decoder" << endl;
    goto end;
  }
  if (avcodec_open2(video_decoder_ctx, video_decoder, nullptr) < 0) {
    cerr << "failed to open video decoder ctx" << endl;
    goto end;
  }

  // 5.fetch frame
  frame = av_frame_alloc();
  packet = av_packet_alloc();
  if (frame == nullptr or packet == nullptr) {
    cerr << "failed to allocate frame or packet" << endl;
    goto end;
  }

  sws_ctx =
      sws_getContext(video_decoder_ctx->width, video_decoder_ctx->height,
                     video_decoder_ctx->pix_fmt, frame_width, frame_height,
                     AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (sws_ctx == nullptr) {
    cerr << "failed to get sws ctx" << endl;
    goto end;
  }

  // 6 show each frame
  while (!quit and av_read_frame(fmt_ctx, packet) == 0) {
    if (packet->stream_index == video_stream_idx) {
      int ret = avcodec_send_packet(video_decoder_ctx, packet);
      if (ret < 0) {
        cerr << "failed to send packet" << endl;
        av_packet_unref(packet);
        break;
      }
      while (true) {
        ret = avcodec_receive_frame(video_decoder_ctx, frame);
        if (ret == AVERROR(EAGAIN) or ret == AVERROR_EOF) {
          av_frame_unref(frame);
          break;
        }
        if (ret < 0) {
          av_frame_unref(frame);
          cerr << "receive frame error" << endl;
          break;
        }
        // show frame
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, data,
                  linesize);
        imshow("camera", img);
        frame_idx++;
        if (frame_idx % 3 == 0) {
          auto res_img = yolo.inference(img);
          imshow("res img", res_img);
        }
        ret = waitKey(1);
        if (ret == 27) {
          quit = true;
          break;
        }
        av_frame_unref(frame);
      }
    }
    av_packet_unref(packet);
  }
  // 7 clean

end:
  if (fmt_ctx != nullptr) avformat_close_input(&fmt_ctx);
  if (options != nullptr) av_dict_free(&options);
  if (video_decoder_ctx != nullptr) {
    avcodec_free_context(&video_decoder_ctx);
  }
  if (frame != nullptr) {
    av_frame_unref(frame);
    av_frame_free(&frame);
  }

  if (packet != nullptr) {
    av_packet_unref(packet);
    av_packet_free(&packet);
  }
}

int main(int argc, char* argv[]) { open_camera(); }