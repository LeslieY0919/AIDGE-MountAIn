#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

extern "C" {
void detector_forward(const int8_t* input, int8_t** out_0, int8_t** out_1, int64_t** out_2, int64_t** out_3, int64_t** out_4);
void classifier_forward(const float* input, float** out_0, int64_t** out_1);
}

struct Detection {
    int class_id;
    float score;
    float x1, y1, x2, y2;
};

static float compute_iou(const Detection& a, const Detection& b) {
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

static void preprocess_detector(const unsigned char* img, int orig_w, int orig_h, std::vector<float>& out_nhwc) {
    constexpr int DST_W = 640;
    constexpr int DST_H = 640;
    out_nhwc.resize(1 * DST_H * DST_W * 3);

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
}

static void preprocess_classifier_crop(const unsigned char* img, int orig_w, int orig_h,
                                       const Detection& det, std::vector<float>& out_nhwc) {
    constexpr int CLS_W = 128;
    constexpr int CLS_H = 128;
    constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};
    out_nhwc.resize(1 * CLS_H * CLS_W * 3);

    int cx1 = std::max(0, static_cast<int>(std::floor(det.x1)));
    int cy1 = std::max(0, static_cast<int>(std::floor(det.y1)));
    int cx2 = std::min(orig_w, static_cast<int>(std::ceil(det.x2)));
    int cy2 = std::min(orig_h, static_cast<int>(std::ceil(det.y2)));
    int crop_w = std::max(1, cx2 - cx1);
    int crop_h = std::max(1, cy2 - cy1);

    const float scale_x = static_cast<float>(crop_w) / static_cast<float>(CLS_W);
    const float scale_y = static_cast<float>(crop_h) / static_cast<float>(CLS_H);

    for (int y = 0; y < CLS_H; ++y) {
        float fy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        int y0 = std::max(0, std::min(crop_h - 1, static_cast<int>(std::floor(fy))));
        int y1 = std::max(0, std::min(crop_h - 1, y0 + 1));
        float wy = std::max(0.0f, std::min(1.0f, fy - static_cast<float>(y0)));

        for (int x = 0; x < CLS_W; ++x) {
            float fx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            int x0 = std::max(0, std::min(crop_w - 1, static_cast<int>(std::floor(fx))));
            int x1 = std::max(0, std::min(crop_w - 1, x0 + 1));
            float wx = std::max(0.0f, std::min(1.0f, fx - static_cast<float>(x0)));

            for (int c = 0; c < 3; ++c) {
                float v00 = static_cast<float>(img[((cy1 + y0) * orig_w + (cx1 + x0)) * 3 + c]);
                float v01 = static_cast<float>(img[((cy1 + y0) * orig_w + (cx1 + x1)) * 3 + c]);
                float v10 = static_cast<float>(img[((cy1 + y1) * orig_w + (cx1 + x0)) * 3 + c]);
                float v11 = static_cast<float>(img[((cy1 + y1) * orig_w + (cx1 + x1)) * 3 + c]);
                float val = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01)
                          + wy * ((1.0f - wx) * v10 + wx * v11);
                out_nhwc[(y * CLS_W + x) * 3 + c] = ((val / 255.0f) - MEAN[c]) / STD[c];
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [conf_thresh]" << std::endl;
        return 1;
    }

    const std::string img_path = argv[1];
    const float conf_thresh = (argc > 2) ? std::strtof(argv[2], nullptr) : 0.25f;
    const float iou_thresh = 0.70f;

    const char* DET_LABELS[3] = {"animal", "person", "vehicle"};
    const char* SPECIES_LABELS[5] = {"blaireau", "chevreuil", "ecureuil", "renard", "sanglier"};
    constexpr int NUM_ANCHORS = 8400;
    constexpr float S_BOXES = 0.1064247691f;
    constexpr float S_SCORES = 0.5669302474f;

    int orig_w = 0, orig_h = 0, channels = 0;
    unsigned char* img = stbi_load(img_path.c_str(), &orig_w, &orig_h, &channels, 3);
    if (!img) {
        std::cerr << "Erreur chargement image : " << img_path << std::endl;
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();

    // 1. Etage Detection (YOLOv10 INT8)
    std::vector<float> det_input;
    preprocess_detector(img, orig_w, orig_h, det_input);
    std::vector<int8_t> det_quant_in(det_input.size());
    for (size_t i = 0; i < det_input.size(); ++i) {
        int q = static_cast<int>(std::round(det_input[i] * 127.0f));
        det_quant_in[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }
    const int8_t* det_in_ptr = det_quant_in.data();

    int8_t* det_out_0 = nullptr;
    int8_t* det_out_1 = nullptr;
    int64_t* det_out_2 = nullptr;
    int64_t* det_out_3 = nullptr;
    int64_t* det_out_4 = nullptr;
    detector_forward(det_in_ptr, &det_out_0, &det_out_1, &det_out_2, &det_out_3, &det_out_4);

    std::vector<Detection> candidates;
    for (int i = 0; i < NUM_ANCHORS; ++i) {
        int best_c = 0;
        float best_logit = static_cast<float>(det_out_1[i]) * S_SCORES;
        for (int c = 1; c < 3; ++c) {
            float l = static_cast<float>(det_out_1[c * NUM_ANCHORS + i]) * S_SCORES;
            if (l > best_logit) {
                best_logit = l;
                best_c = c;
            }
        }
        float best_score = 1.0f / (1.0f + std::exp(-best_logit));
        if (best_score < conf_thresh) continue;

        float dist[4];
        for (int side = 0; side < 4; ++side) {
            float max_v = static_cast<float>(det_out_0[(side * 16) * NUM_ANCHORS + i]) * S_BOXES;
            for (int b = 1; b < 16; ++b) {
                float v = static_cast<float>(det_out_0[(side * 16 + b) * NUM_ANCHORS + i]) * S_BOXES;
                if (v > max_v) max_v = v;
            }
            float sum_e = 0.0f, acc = 0.0f;
            for (int b = 0; b < 16; ++b) {
                float v = static_cast<float>(det_out_0[(side * 16 + b) * NUM_ANCHORS + i]) * S_BOXES;
                float e = std::exp(v - max_v);
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

    // 2. Etage Classification (ResNet-34 C++) sur chaque animal detecte
    std::vector<float> cls_input;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Image: " << img_path << " (" << orig_w << "x" << orig_h << ") -> "
              << final_dets.size() << " detection(s):" << std::endl;

    for (size_t k = 0; k < final_dets.size(); ++k) {
        const auto& d = final_dets[k];
        std::cout << "  #" << k << " [" << DET_LABELS[d.class_id] << " " << (d.score * 100.0f) << "%]"
                  << " bbox=[" << static_cast<int>(d.x1) << ", " << static_cast<int>(d.y1)
                  << ", " << static_cast<int>(d.x2) << ", " << static_cast<int>(d.y2) << "]";

        if (d.class_id == 0) {
            preprocess_classifier_crop(img, orig_w, orig_h, d, cls_input);
            float* cls_out_0 = nullptr;
            int64_t* cls_out_1 = nullptr;
            classifier_forward(cls_input.data(), &cls_out_0, &cls_out_1);

            float max_l = cls_out_0[0];
            for (int c = 1; c < 5; ++c) {
                if (cls_out_0[c] > max_l) max_l = cls_out_0[c];
            }
            float sum_exp = 0.0f;
            float probs[5];
            for (int c = 0; c < 5; ++c) {
                probs[c] = std::exp(cls_out_0[c] - max_l);
                sum_exp += probs[c];
            }
            int best_sp = 0;
            for (int c = 0; c < 5; ++c) {
                probs[c] /= sum_exp;
                if (probs[c] > probs[best_sp]) best_sp = c;
            }
            std::cout << " -> ESPECE: " << SPECIES_LABELS[best_sp]
                      << " (" << (probs[best_sp] * 100.0f) << "%)";
        }
        std::cout << std::endl;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Temps total pipeline C++ (preprocess + YOLOv10 + ResNet34) : " << ms << " ms" << std::endl;

    stbi_image_free(img);
    return 0;
}
