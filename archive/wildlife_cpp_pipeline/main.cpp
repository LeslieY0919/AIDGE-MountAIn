#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma GCC diagnostic pop

extern "C" {
void det_model_forward(const float* input_0,
                       float** concat_14_out,
                       float** concat_15_out,
                       int64_t** mp0_out,
                       int64_t** mp1_out,
                       int64_t** mp2_out);

void cls_model_forward(const float* input_0,
                       float** fc_0_out);
}

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
    float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float denom = area_a + area_b - inter;
    return (denom > 0.0f) ? (inter / denom) : 0.0f;
}

// Reproduit PIL.Image.Resampling.BILINEAR (filtre triangulaire separable anti-aliasing)
static void resize_rgb_nhwc(const unsigned char* img, int orig_w, int orig_h,
                            int roi_x0, int roi_y0, int roi_w, int roi_h,
                            int dst_w, int dst_h,
                            const float mean[3], const float inv_std[3],
                            float* out_nhwc)
{
    roi_x0 = std::max(0, std::min(orig_w - 1, roi_x0));
    roi_y0 = std::max(0, std::min(orig_h - 1, roi_y0));
    roi_w = std::max(1, std::min(orig_w - roi_x0, roi_w));
    roi_h = std::max(1, std::min(orig_h - roi_y0, roi_h));

    const float scale_x = static_cast<float>(roi_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(roi_h) / static_cast<float>(dst_h);
    const float support_x = std::max(1.0f, scale_x);
    const float support_y = std::max(1.0f, scale_y);
    const float inv_sup_x = 1.0f / support_x;
    const float inv_sup_y = 1.0f / support_y;

    // Passe 1 : horizontale (roi_h x dst_w x 3), arrondie en [0, 255] comme libImaging
    std::vector<float> tmp(static_cast<size_t>(roi_h) * static_cast<size_t>(dst_w) * 3);

    for (int dx = 0; dx < dst_w; ++dx) {
        const float center = (static_cast<float>(dx) + 0.5f) * scale_x;
        const int xmin = std::max(0, static_cast<int>(center - support_x + 0.5f));
        const int xmax = std::min(roi_w, static_cast<int>(center + support_x + 0.5f));

        std::vector<float> wx(static_cast<size_t>(std::max(1, xmax - xmin)));
        float wsum = 0.0f;
        for (int sx = xmin; sx < xmax; ++sx) {
            float d = std::fabs((static_cast<float>(sx) + 0.5f - center) * inv_sup_x);
            float w = std::max(0.0f, 1.0f - d);
            wx[static_cast<size_t>(sx - xmin)] = w;
            wsum += w;
        }
        const float norm = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;
        for (float& w : wx) w *= norm;

        for (int sy = 0; sy < roi_h; ++sy) {
            const unsigned char* src_row = img + ((roi_y0 + sy) * orig_w + roi_x0) * 3;
            float* dst_px = tmp.data() + (sy * dst_w + dx) * 3;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
            for (int sx = xmin; sx < xmax; ++sx) {
                const float w = wx[static_cast<size_t>(sx - xmin)];
                const unsigned char* p = src_row + sx * 3;
                acc0 += w * static_cast<float>(p[0]);
                acc1 += w * static_cast<float>(p[1]);
                acc2 += w * static_cast<float>(p[2]);
            }
            dst_px[0] = std::min(255.0f, std::max(0.0f, std::round(acc0)));
            dst_px[1] = std::min(255.0f, std::max(0.0f, std::round(acc1)));
            dst_px[2] = std::min(255.0f, std::max(0.0f, std::round(acc2)));
        }
    }

    // Passe 2 : verticale (dst_h x dst_w x 3) + normalisation ImageNet
    for (int dy = 0; dy < dst_h; ++dy) {
        const float center = (static_cast<float>(dy) + 0.5f) * scale_y;
        const int ymin = std::max(0, static_cast<int>(center - support_y + 0.5f));
        const int ymax = std::min(roi_h, static_cast<int>(center + support_y + 0.5f));

        std::vector<float> wy(static_cast<size_t>(std::max(1, ymax - ymin)));
        float wsum = 0.0f;
        for (int sy = ymin; sy < ymax; ++sy) {
            float d = std::fabs((static_cast<float>(sy) + 0.5f - center) * inv_sup_y);
            float w = std::max(0.0f, 1.0f - d);
            wy[static_cast<size_t>(sy - ymin)] = w;
            wsum += w;
        }
        const float norm = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;
        for (float& w : wy) w *= norm;

        for (int dx = 0; dx < dst_w; ++dx) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
            for (int sy = ymin; sy < ymax; ++sy) {
                const float w = wy[static_cast<size_t>(sy - ymin)];
                const float* p = tmp.data() + (sy * dst_w + dx) * 3;
                acc0 += w * p[0];
                acc1 += w * p[1];
                acc2 += w * p[2];
            }
            const float v0 = std::min(255.0f, std::max(0.0f, std::round(acc0)));
            const float v1 = std::min(255.0f, std::max(0.0f, std::round(acc1)));
            const float v2 = std::min(255.0f, std::max(0.0f, std::round(acc2)));

            float* out_px = out_nhwc + (dy * dst_w + dx) * 3;
            out_px[0] = (v0 / 255.0f - mean[0]) * inv_std[0];
            out_px[1] = (v1 / 255.0f - mean[1]) * inv_std[1];
            out_px[2] = (v2 / 255.0f - mean[2]) * inv_std[2];
        }
    }
}

static std::vector<Detection> decode_and_nms(const float* boxes_ptr,
                                             const float* scores_ptr,
                                             int orig_w, int orig_h,
                                             float conf_thresh,
                                             float iou_thresh)
{
    constexpr int NUM_ANCHORS = 8400;
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
    return final_dets;
}

int main(int argc, char** argv)
{
    using Clock = std::chrono::steady_clock;

    const std::string image_path = (argc > 1) ? argv[1] : "../samples/test_wildlife_1mp.jpg";
    const float conf_thresh = (argc > 2) ? std::strtof(argv[2], nullptr) : 0.25f;
    const int n_iters = (argc > 3) ? std::max(1, std::atoi(argv[3])) : 10;
    const float iou_thresh = 0.70f;

    const char* DET_LABELS[3] = {"animal", "person", "vehicle"};
    const char* SPECIES_LABELS[5] = {"blaireau", "chevreuil", "ecureuil", "renard", "sanglier"};

    const float DET_MEAN[3] = {0.0f, 0.0f, 0.0f};
    const float DET_INV_STD[3] = {1.0f, 1.0f, 1.0f};
    const float CLS_MEAN[3] = {0.485f, 0.456f, 0.406f};
    const float CLS_INV_STD[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

    int orig_w = 0, orig_h = 0, channels = 0;
    unsigned char* img = stbi_load(image_path.c_str(), &orig_w, &orig_h, &channels, 3);
    if (!img) {
        std::cerr << "Erreur : impossible de charger " << image_path << std::endl;
        return 1;
    }

    std::vector<float> det_input(1 * 640 * 640 * 3);
    std::vector<float> cls_input(1 * 128 * 128 * 3);

    double total_pre_det_ms = 0.0;
    double total_det_inf_ms = 0.0;
    double total_post_det_ms = 0.0;
    double total_cls_inf_ms = 0.0;
    int total_cls_calls = 0;

    std::vector<Detection> last_dets;
    std::vector<int> last_species_ids;
    std::vector<float> last_species_confs;

    for (int iter = 0; iter < n_iters; ++iter) {
        auto t0 = Clock::now();
        resize_rgb_nhwc(img, orig_w, orig_h, 0, 0, orig_w, orig_h, 640, 640, DET_MEAN, DET_INV_STD, det_input.data());
        auto t1 = Clock::now();

        float* boxes_ptr = nullptr;
        float* scores_ptr = nullptr;
        int64_t* idx0 = nullptr;
        int64_t* idx1 = nullptr;
        int64_t* idx2 = nullptr;
        det_model_forward(det_input.data(), &boxes_ptr, &scores_ptr, &idx0, &idx1, &idx2);
        auto t2 = Clock::now();

        auto dets = decode_and_nms(boxes_ptr, scores_ptr, orig_w, orig_h, conf_thresh, iou_thresh);
        auto t3 = Clock::now();

        std::vector<int> sp_ids(dets.size(), -1);
        std::vector<float> sp_confs(dets.size(), 0.0f);

        for (size_t d = 0; d < dets.size(); ++d) {
            if (dets[d].class_id != 0) continue;

            int ix1 = std::max(0, static_cast<int>(std::round(dets[d].x1)));
            int iy1 = std::max(0, static_cast<int>(std::round(dets[d].y1)));
            int ix2 = std::min(orig_w, std::max(ix1 + 1, static_cast<int>(std::round(dets[d].x2))));
            int iy2 = std::min(orig_h, std::max(iy1 + 1, static_cast<int>(std::round(dets[d].y2))));

            auto tc0 = Clock::now();
            resize_rgb_nhwc(img, orig_w, orig_h, ix1, iy1, ix2 - ix1, iy2 - iy1, 128, 128, CLS_MEAN, CLS_INV_STD, cls_input.data());
            float* logits_ptr = nullptr;
            cls_model_forward(cls_input.data(), &logits_ptr);
            auto tc1 = Clock::now();

            total_cls_inf_ms += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
            total_cls_calls++;

            float max_l = logits_ptr[0];
            for (int k = 1; k < 5; ++k) {
                if (logits_ptr[k] > max_l) max_l = logits_ptr[k];
            }
            float sum_exp = 0.0f;
            float probs[5];
            for (int k = 0; k < 5; ++k) {
                probs[k] = std::exp(logits_ptr[k] - max_l);
                sum_exp += probs[k];
            }
            int best_k = 0;
            for (int k = 0; k < 5; ++k) {
                probs[k] /= sum_exp;
                if (probs[k] > probs[best_k]) best_k = k;
            }
            sp_ids[d] = best_k;
            sp_confs[d] = probs[best_k];
        }

        total_pre_det_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_det_inf_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_post_det_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        if (iter == n_iters - 1) {
            last_dets = std::move(dets);
            last_species_ids = std::move(sp_ids);
            last_species_confs = std::move(sp_confs);
        }
    }

    stbi_image_free(img);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Pipeline C++ Autonome (Aidge) ===" << std::endl;
    std::cout << "Image : " << image_path << " (" << orig_w << "x" << orig_h << ")" << std::endl;
    for (size_t i = 0; i < last_dets.size(); ++i) {
        const auto& d = last_dets[i];
        std::cout << "  #" << i << " [" << DET_LABELS[d.class_id] << " " << (d.score * 100.0f) << "%] "
                  << "bbox=[" << static_cast<int>(std::round(d.x1)) << ", "
                  << static_cast<int>(std::round(d.y1)) << ", "
                  << static_cast<int>(std::round(d.x2)) << ", "
                  << static_cast<int>(std::round(d.y2)) << "]";
        if (last_species_ids[i] >= 0) {
            std::cout << " -> ESPECE: " << SPECIES_LABELS[last_species_ids[i]]
                      << " (" << (last_species_confs[i] * 100.0f) << "%)";
        }
        std::cout << std::endl;
    }

    double avg_pre = total_pre_det_ms / n_iters;
    double avg_det = total_det_inf_ms / n_iters;
    double avg_post = total_post_det_ms / n_iters;
    double avg_cls = (total_cls_calls > 0) ? (total_cls_inf_ms / total_cls_calls) : 0.0;
    double avg_total = avg_pre + avg_det + avg_post + (total_cls_inf_ms / n_iters);

    std::cout << "\n=== Latences moyennes sur " << n_iters << " iterations ===" << std::endl;
    std::cout << "  Pretraitement 640x640  : " << avg_pre << " ms" << std::endl;
    std::cout << "  Inference Detecteur    : " << avg_det << " ms" << std::endl;
    std::cout << "  Decodage DFL + NMS     : " << avg_post << " ms" << std::endl;
    std::cout << "  Crop + Classifieur     : " << avg_cls << " ms / animal" << std::endl;
    std::cout << "  TOTAL PIPELINE         : " << avg_total << " ms (" << (1000.0 / avg_total) << " FPS)" << std::endl;

    return 0;
}
