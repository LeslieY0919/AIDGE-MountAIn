#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "forward.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Detection {
    int class_id;
    float score;
    float x1, y1, x2, y2;
};

static float compute_iou(const Detection& a, const Detection& b)
{
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float denom = area_a + area_b - inter;
    return (denom > 0.0f) ? (inter / denom) : 0.0f;
}

static bool load_and_preprocess(const std::string& path, std::vector<float>& out_nhwc, int& orig_w, int& orig_h)
{
    constexpr int DST_W = 640;
    constexpr int DST_H = 640;
    out_nhwc.resize(1 * DST_H * DST_W * 3);

    if (path.size() >= 4 && path.substr(path.size() - 4) == ".bin") {
        std::ifstream in_file(path, std::ios::binary);
        if (!in_file) return false;
        in_file.read(reinterpret_cast<char*>(out_nhwc.data()), out_nhwc.size() * sizeof(float));
        orig_w = DST_W;
        orig_h = DST_H;
        return static_cast<bool>(in_file);
    }

    int channels = 0;
    unsigned char* img = stbi_load(path.c_str(), &orig_w, &orig_h, &channels, 3);
    if (!img) return false;

    const float scale_x = static_cast<float>(orig_w) / static_cast<float>(DST_W);
    const float scale_y = static_cast<float>(orig_h) / static_cast<float>(DST_H);

    for (int y = 0; y < DST_H; ++y) {
        float fy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        int y0 = std::max(0, std::min(orig_h - 1, static_cast<int>(std::floor(fy))));
        int y1 = std::max(0, std::min(orig_h - 1, y0 + 1));
        float wy = std::max(0.0f, std::min(1.0f, fy - static_cast<float>(y0)));

        for (int x = 0; x < DST_W; ++x) {
            float fx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            int x0 = std::max(0, std::min(orig_w - 1, static_cast<int>(std::floor(fx))));
            int x1 = std::max(0, std::min(orig_w - 1, x0 + 1));
            float wx = std::max(0.0f, std::min(1.0f, fx - static_cast<float>(x0)));

            for (int c = 0; c < 3; ++c) {
                float v00 = static_cast<float>(img[(y0 * orig_w + x0) * 3 + c]);
                float v01 = static_cast<float>(img[(y0 * orig_w + x1) * 3 + c]);
                float v10 = static_cast<float>(img[(y1 * orig_w + x0) * 3 + c]);
                float v11 = static_cast<float>(img[(y1 * orig_w + x1) * 3 + c]);
                float val = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01)
                          + wy * ((1.0f - wx) * v10 + wx * v11);
                out_nhwc[(y * DST_W + x) * 3 + c] = val / 255.0f;
            }
        }
    }

    stbi_image_free(img);
    return true;
}

int main(int argc, char** argv)
{
    const std::string input_path = (argc > 1) ? argv[1] : "test_wildlife_1mp.jpg";
    const float conf_thresh = (argc > 2) ? std::strtof(argv[2], nullptr) : 0.25f;
    const float iou_thresh = 0.70f;
    const char* LABELS[3] = {"animal", "person", "vehicle"};
    constexpr int NUM_ANCHORS = 8400;

    std::vector<float> input_data;
    int orig_w = 640, orig_h = 640;
    if (!load_and_preprocess(input_path, input_data, orig_w, orig_h)) {
        std::cerr << "Erreur : impossible de charger " << input_path << std::endl;
        return 1;
    }

    float* boxes_ptr = nullptr;
    float* scores_ptr = nullptr;
    int64_t* idx0 = nullptr;
    int64_t* idx1 = nullptr;
    int64_t* idx2 = nullptr;

    model_forward(input_data.data(), &boxes_ptr, &scores_ptr, &idx0, &idx1, &idx2);

    std::vector<Detection> candidates;
    for (int i = 0; i < NUM_ANCHORS; ++i) {
        int best_c = 0;
        float best_logit = scores_ptr[i];
        for (int c = 1; c < 3; ++c) {
            float l = scores_ptr[c * NUM_ANCHORS + i];
            if (l > best_logit) {
                best_logit = l;
                best_c = c;
            }
        }
        float best_score = 1.0f / (1.0f + std::exp(-best_logit));
        if (best_score < conf_thresh) continue;

        float dist[4];
        for (int side = 0; side < 4; ++side) {
            float max_v = boxes_ptr[(side * 16) * NUM_ANCHORS + i];
            for (int b = 1; b < 16; ++b) {
                float v = boxes_ptr[(side * 16 + b) * NUM_ANCHORS + i];
                if (v > max_v) max_v = v;
            }
            float sum_e = 0.0f;
            float acc = 0.0f;
            for (int b = 0; b < 16; ++b) {
                float e = std::exp(boxes_ptr[(side * 16 + b) * NUM_ANCHORS + i] - max_v);
                sum_e += e;
                acc += e * static_cast<float>(b);
            }
            dist[side] = acc / sum_e;
        }

        float gx = 0.0f, gy = 0.0f, stride = 8.0f;
        if (i < 6400) {
            gx = static_cast<float>(i % 80) + 0.5f;
            gy = static_cast<float>(i / 80) + 0.5f;
            stride = 8.0f;
        } else if (i < 8000) {
            int idx = i - 6400;
            gx = static_cast<float>(idx % 40) + 0.5f;
            gy = static_cast<float>(idx / 40) + 0.5f;
            stride = 16.0f;
        } else {
            int idx = i - 8000;
            gx = static_cast<float>(idx % 20) + 0.5f;
            gy = static_cast<float>(idx / 20) + 0.5f;
            stride = 32.0f;
        }

        float x1 = std::max(0.0f, std::min(640.0f, (gx - dist[0]) * stride)) * (static_cast<float>(orig_w) / 640.0f);
        float y1 = std::max(0.0f, std::min(640.0f, (gy - dist[1]) * stride)) * (static_cast<float>(orig_h) / 640.0f);
        float x2 = std::max(0.0f, std::min(640.0f, (gx + dist[2]) * stride)) * (static_cast<float>(orig_w) / 640.0f);
        float y2 = std::max(0.0f, std::min(640.0f, (gy + dist[3]) * stride)) * (static_cast<float>(orig_h) / 640.0f);

        candidates.push_back({best_c, best_score, x1, y1, x2, y2});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> final_dets;
    for (const auto& cand : candidates) {
        bool suppressed = false;
        for (const auto& kept : final_dets) {
            if (cand.class_id == kept.class_id && compute_iou(cand, kept) > iou_thresh) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) final_dets.push_back(cand);
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Image: " << input_path << " (" << orig_w << "x" << orig_h << ") -> Detections (" << final_dets.size() << "):" << std::endl;
    for (const auto& d : final_dets) {
        std::cout << "  [" << LABELS[d.class_id] << "] conf=" << d.score
                  << " bbox=[" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]"
                  << std::endl;
    }
    return 0;
}
