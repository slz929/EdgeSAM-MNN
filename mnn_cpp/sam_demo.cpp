#include <stdio.h>
#include <chrono>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <cv/cv.hpp>

#include <opencv2/opencv.hpp>

#include<iostream>

using namespace MNN;
using namespace MNN::Express;
// using namespace MNN::CV;

void get_grid_points(std::vector<float>& points_xy_vec, int n_per_side)
{
    float offset = 1.f / (2 * n_per_side);
    
    float start = offset;
    float end = 1 - offset;
    float step = (end - start) / (n_per_side - 1);

    std::vector<float> points_one_side;
    for (int i = 0; i < n_per_side; ++i) {
        points_one_side.push_back(start + i * step);
    }

    points_xy_vec.resize(n_per_side * n_per_side * 2);
    for (int i = 0; i < n_per_side; ++i) {
        for (int j = 0; j < n_per_side; ++j) {
            points_xy_vec[i * n_per_side * 2 + 2 * j + 0] = points_one_side[j];
            points_xy_vec[i * n_per_side * 2 + 2 * j + 1] = points_one_side[i];
        }
    }
}


int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./sam_demo.out embed.mnn sam.mnn input.jpg [forwardType] [precision] [thread]\n");
        return 0;
    }
    int thread = 4;
    int precision = 0;
    int forwardType = MNN_FORWARD_CPU;
    if (argc >= 5) {
        forwardType = atoi(argv[4]);
    }
    if (argc >= 6) {
        precision = atoi(argv[5]);
    }
    if (argc >= 7) {
        thread = atoi(argv[6]);
    }
    float mask_threshold = 0;
    MNN::ScheduleConfig sConfig;
    sConfig.type = static_cast<MNNForwardType>(forwardType);
    sConfig.numThread = thread;
    BackendConfig bConfig;
    bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    sConfig.backendConfig = &bConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(sConfig));
    if(rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
        return 0;
    }
    // rtmgr->setCache(".cachefile");
    std::shared_ptr<Module> embed(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));
    std::shared_ptr<Module> sam(Module::load(
        {"point_coords", "point_labels", "image_embeddings"},
        {"scores" , "masks"}, argv[2], rtmgr));
    auto image = MNN::CV::imread(argv[3]);

    // image= MNN::CV::resize(image, )
    // 1. preprocess
    auto dims = image->getInfo()->dim;
    int origin_h = dims[0];
    int origin_w = dims[1];
    int length = 1024;
    int new_h, new_w;
    if (origin_h > origin_w) {
        new_w = round(origin_w * (float)length / origin_h);
        new_h = length;
    } else {
        new_h = round(origin_h * (float)length / origin_w);
        new_w = length;
    }
    float scale_w = (float)new_w / origin_w;
    float scale_h = (float)new_h / origin_h;
    auto input_var = MNN::CV::resize(image, MNN::CV::Size(new_w, new_h), 0, 0, MNN::CV::INTER_LINEAR, -1, {123.675, 116.28, 103.53}, {1/58.395, 1/57.12, 1/57.375});
    std::vector<int> padvals { 0, length - new_h, 0, length - new_w, 0, 0 };
    auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
    input_var = _Pad(input_var, pads, CONSTANT);
    input_var = _Unsqueeze(input_var, {0});
    // 2. image embedding
    for(int i =0; i< input_var->getInfo()->dim.size(); i++)
        std::cout<< input_var->getInfo()->dim[i]<<" ";
    std::cout<<"\n";

    input_var = _Convert(input_var, NC4HW4);
    for(int i =0; i< input_var->getInfo()->dim.size(); i++)
        std::cout<< input_var->getInfo()->dim[i]<<" ";
    std::cout<<"\n";


    std::cout<<"version="<< embed->getInfo()->version<<"\n";
    std::cout<<"inputNames=";
    for(auto s : embed->getInfo()->inputNames)
        std::cout<<s<<" ";
    std::cout<<"\n";
    std::cout<<"outputNames=";
    for(auto s : embed->getInfo()->outputNames)
        std::cout<<s<<" ";
    std::cout<<"\n";

    auto st = std::chrono::system_clock::now();
    auto outputs = embed->onForward({input_var});
    auto et = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(et - st);
    printf("# 1. embedding times: %f ms\n", duration.count() * 1e-3);

    auto image_embedding = _Convert(outputs[0], NCHW);

    // 3. segment
    auto build_input = [](std::vector<float> data, std::vector<int> shape) {
        return _Const(static_cast<void*>(data.data()), shape, NCHW, halide_type_of<float>());
    };
    // build inputs
    std::vector<float> points;
    std::vector<float> labels;
    int n_per_side= 3;
    get_grid_points(points, n_per_side);
    float region_ratio= 0.5; 
    for(int i = 0; i < n_per_side; ++i) {
        for(int j = 0; j < n_per_side; ++j) {
            int x= i * n_per_side * 2 + 2 * j;
            points[x] = points[x] * origin_w * region_ratio + origin_w * region_ratio * 0.5;
            points[x+ 1] = points[x + 1] * origin_h * region_ratio + origin_h * region_ratio * 0.5;
        }
    }
    std::vector<float> scale_points;
    for(int i = 0; i < n_per_side; ++i) {
        for(int j = 0; j < n_per_side; ++j) {
            int x= i * n_per_side * 2 + 2 * j;
            scale_points.push_back(points[x] * scale_w);
            scale_points.push_back(points[x + 1] * scale_h);
            scale_points.push_back(0);
            scale_points.push_back(0);
            labels.push_back(1);
            labels.push_back(-1);
        }
    }

    auto point_coords = build_input(scale_points, {1, 2* n_per_side*n_per_side , 2});
    
    auto point_labels = build_input(labels, {1, 2 *n_per_side*n_per_side} );
    auto orig_im_size = build_input({static_cast<float>(origin_h), static_cast<float>(origin_w)}, {2});
    // auto has_mask_input = build_input({0}, {1});
    // std::vector<float> zeros(256*256, 0.f);
    // auto mask_input = build_input(zeros, {1, 1, 256, 256});
    st = std::chrono::system_clock::now();
    auto output_vars = sam->onForward({point_coords, point_labels, image_embedding});
    et = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(et - st);
    printf("# 2. segment times: %f ms\n", duration.count() * 1e-3);
    auto masks = _Convert(output_vars[1], NCHW);
    auto scores = _Convert(output_vars[0], NCHW);

    std::cout<<"masks:";
    for(int i =0; i< masks->getInfo()->dim.size(); i++)
        std::cout<< masks->getInfo()->dim[i]<<" ";
    std::cout<<"\n";

    scores= _Squeeze(scores, {0});
    std::vector<std::pair<float, int>> scores_vec;
    auto outputPtr = scores->readMap<float>();
    auto outputSize = scores->getInfo()->size;
    for (int i=0; i<outputSize; ++i) {
        scores_vec.push_back(std::pair<float, int>(outputPtr[i], i));
    }
    std::sort(scores_vec.begin(), scores_vec.end(), std::greater<std::pair<float, int>>());

    float pred_iou_thresh= 0.8f;
    if (scores_vec[0].first > pred_iou_thresh) {
        int ch = scores_vec[0].second;
        masks = _Gather(_Squeeze(masks, {0}), _Scalar<int>(ch));
        int h= masks->getInfo()->dim[0];
        int w= masks->getInfo()->dim[1];
        // resize to (length, length)
        cv::Mat mask_cv(h, w, CV_32F);; 
        const float* dataPtr= masks->readMap<float>();
        memcpy(mask_cv.data, dataPtr, h*w* sizeof(float));
        cv::resize(mask_cv, mask_cv, cv::Size(length, length));
        // masks = _Convert(masks, NC4HW4);
        // auto masks_out= _Resize(masks, (float)(length)/h, (float)(length)/w);
        // masks_out = _Convert(masks_out, NCHW);
        // masks= MNN::CV::resize(masks, MNN::CV::Size(length, length), MNN::CV::BILINEAR);

        // crop (new_w, new_h)
        cv::Rect roi(0, 0, new_w, new_h);
        cv::Mat mask_crop = mask_cv(roi);

        // resize to (origin_w, origin_h)
        cv::resize(mask_crop, mask_crop, cv::Size(origin_w, origin_h));
        auto masks_out = _Input({origin_h, origin_w}, NHWC, halide_type_of<float>());
        auto dataPtr_write= masks_out->writeMap<float>();
        memcpy(dataPtr_write, mask_crop.data, origin_h* origin_w* sizeof(float));
        masks= masks_out;

        // outputSize = masks->getInfo()->size;
        // outputPtr = masks->readMap<float>();
        // for (int i=0; i<5; ++i) {
        //     printf("%d: %f\n", i, outputPtr[i]);
        // }
        
        masks = _Greater(masks, _Scalar(mask_threshold));
        masks = _Reshape(masks, {origin_h, origin_w, 1});
        std::vector<int> color_vec {255, 153, 0};
        auto color = _Const(static_cast<void*>(color_vec.data()), {1, 1, 3}, NCHW, halide_type_of<int>());
        float w1= 0.6;
        float w2= 0.4;    
        auto alpha = _Const(static_cast<void*>(&w1), {1, 1, 1}, NCHW, halide_type_of<float>());     
        auto beta = _Const(static_cast<void*>(&w2), {1, 1, 1}, NCHW, halide_type_of<float>());    
        image = _Cast<uint8_t>(_Cast<int>(_Cast<float>(image) * alpha) + _Cast<int>(_Cast<float>(masks) * _Cast<float>(color) * beta));
        for (int i = 0; i < points.size() / 2; i++) {
            float x = points[2 * i];
            float y = points[2 * i + 1];
            MNN::CV::circle(image, {x, y}, 10, {0, 0, 255}, 5);
        }
    }

    // 4. postprocess: draw mask and point
        if (MNN::CV::imwrite("res_.jpg", image)) {
            MNN_PRINT("result image write to `res.jpg`.\n");
        }
    // rtmgr->updateCache();
    return 0;
}
