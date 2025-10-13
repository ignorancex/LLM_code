#include "xfeat.h"
#include "onnx_helper.h"

XFeat::XFeat()
{
	std::string xfeatModelPath = "/home/luo/cyq/OFVTR/VINS/src/VINS-Fusion/loop_fusion/model/xfeat_800.onnx";
    std::string xfeatlightgluePath = "/home/luo/cyq/OFVTR/VINS/src/VINS-Fusion/loop_fusion/model/xfeat_lightglue.onnx";
	const ORTCHAR_T* ortXfeatModelPath = xfeatModelPath.c_str();
    const ORTCHAR_T* ortXfeatLightgluePath = xfeatlightgluePath.c_str();

	env_ = Ort::Env{ OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL, "xfeat_demo" };  //  ORT_LOGGING_LEVEL_VERBOSE, ORT_LOGGING_LEVEL_FATAL

	std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
	// std::cout << "All available accelerators:" << std::endl;
	// for (int i = 0; i < availableProviders.size(); i++)
	// {
	// 	std::cout << "  " << i + 1 << ". " << availableProviders[i] << std::endl;
	// }
	// init sessions
	initOrtSession(env_, xfeatSession_, xfeatModelPath, gpuId_);
    initOrtSession(env_, matchingSession_, xfeatlightgluePath, gpuId_);
}

XFeat::~XFeat()
{
	env_.release();
	xfeatSession_.release();
    matchingSession_.release();
}

int XFeat::detectAndCompute(const cv::Mat& image, cv::Mat& mkpts, cv::Mat& feats, cv::Mat& sc)
{
	// 图片预处理
	cv::Mat preProcessedImage = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
	int stride = preProcessedImage.rows * preProcessedImage.cols;
    // printf("将图片形状转换为模型输入要求的形状\n");
// #pragma omp parallel for 
    // 将图片形状转换为模型输入要求的形状
	for (int i = 0; i < stride; i++) // HWC -> CHW, BGR -> RGB
	{
		*((float*)preProcessedImage.data + i) = (float)*(image.data + i * 3 + 2);
		*((float*)preProcessedImage.data + i + stride) = (float)*(image.data + i * 3 + 1);
		*((float*)preProcessedImage.data + i + stride * 2) = (float)*(image.data + i * 3);
	}

	// printf("// 创造输入的tensor\n");
	int64_t input_size = preProcessedImage.rows * preProcessedImage.cols * 3;
	std::vector<int64_t> input_node_dims = { 1, 3, preProcessedImage.rows , preProcessedImage.cols };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)(preProcessedImage.data), input_size, input_node_dims.data(), input_node_dims.size());
	assert(input_tensor.IsTensor());


	// 运行模型
    // output_tensors中存储的分别是特征点，描述子，得分矩阵
    // printf("//运行模型\n");
	auto output_tensors =
		xfeatSession_.Run(Ort::RunOptions{ nullptr }, xfeatInputNames.data(),
			&input_tensor, xfeatInputNames.size(), xfeatOutputNames.data(), xfeatOutputNames.size());
	assert(output_tensors.size() == xfeatOutputNames.size() && output_tensors.front().IsTensor());

	// 得到输出特征点
    // printf("//1得到输出特征点\n");
	auto mkptsShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	int dim1 = static_cast<int>(mkptsShape[0]); 
	int dim2 = static_cast<int>(mkptsShape[1]); 
	int dim3 = static_cast<int>(mkptsShape[2]); // 2
	float* mkptsDataPtr = output_tensors[0].GetTensorMutableData<float>();
	// To cv::Mat
	cv::Mat raw_mkpts = cv::Mat(dim1, dim2, CV_32FC(dim3), mkptsDataPtr).clone();
    mkpts = raw_mkpts.reshape(1, raw_mkpts.size[1]);
    // printf("//2得到输出特征点\n");
	auto featsShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
	dim1 = static_cast<int>(featsShape[0]); 
	dim2 = static_cast<int>(featsShape[1]); 
	dim3 = static_cast<int>(featsShape[2]); // 64
	float* featsDataPtr = output_tensors[1].GetTensorMutableData<float>();
	cv::Mat raw_feats = cv::Mat(dim1, dim2, CV_32FC(dim3), featsDataPtr).clone();
    feats = raw_feats.reshape(1, raw_feats.size[1]);
    // feats = cv::Mat(dim1, dim2, CV_32FC(dim3), featsDataPtr).clone();
//    printf("//3得到输出特征点\n");
	auto scShape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
	dim1 = static_cast<int>(scShape[0]); 
	dim2 = static_cast<int>(scShape[1]); 
	float* scDataPtr = output_tensors[2].GetTensorMutableData<float>();
	sc = cv::Mat(dim1, dim2, CV_32F, scDataPtr).clone();
    // printf("//4得到输出特征点\n");
	return 0;
}

cv::Mat sampleMat(const cv::Mat& src, int rowStep, int colStep) {
    CV_Assert(rowStep > 0 && colStep > 0); // 确保步长是正数
 
    int newRows = (src.rows + rowStep - 1) / rowStep; // 计算新的行数（向上取整）
    int newCols = (src.cols + colStep - 1) / colStep; // 计算新的列数（向上取整）
 
    cv::Mat sampled(newRows, newCols, src.type()); // 创建新的采样后的矩阵
 
    for (int i = 0; i < newRows; ++i) {
        for (int j = 0; j < newCols; ++j) {
            // 计算原始矩阵中的对应位置
            int origRow = i * rowStep;
            int origCol = j * colStep;
 
            // 复制元素到新的矩阵中
            sampled.at<uchar>(i, j) = src.at<uchar>(origRow, origCol); // 假设是3通道图像
            // 如果是单通道图像，则使用：sampled.at<uchar>(i, j) = src.at<uchar>(origRow, origCol);
        }
    }
 
    return sampled;
}

int XFeat::match(const cv::Mat& mkpts0, const cv::Mat& feats0, const cv::Mat& mkpts1, const cv::Mat& feats1, std::vector<cv::DMatch>& matches, float minScore,
                    std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2){
    // std::vector<cv::Point2f> points_1;
    // std::vector<cv::Point2f> points_2;
     auto start = std::chrono::high_resolution_clock::now();
    cv::Mat mkpts0_reshaped = mkpts0.reshape(1, mkpts0.size[1]);
    cv::Mat mkpts1_reshaped = mkpts1.reshape(1, mkpts1.size[1]);
    cv::Mat feats0_reshaped = feats0.reshape(1, feats0.size[1]);
    cv::Mat feats1_reshaped = feats1.reshape(1, feats1.size[1]);
    // std::cout<<"mkpts0_reshaped size "<<mkpts0_reshaped.rows<<" "<<mkpts0_reshaped.cols<<" "<<mkpts0_reshaped.channels()<<std::endl;
    // std::cout<<"feats0_reshaped size "<<feats0_reshaped.rows<<" "<<feats0_reshaped.cols<<" "<<feats0_reshaped.channels()<<std::endl;
    
	// cv::Mat mkpts0_sampled = sampleMat(mkpts0_reshaped, 3, 1);
	// cv::Mat feats0_sampled = sampleMat( feats0_reshaped, 3, 1);
	// cv::Mat mkpts1_sampled = sampleMat(mkpts1_reshaped, 3, 1);
	// cv::Mat feats1_sampled = sampleMat( feats1_reshaped, 3, 1);
	// std::cout<<"mkpts0_sampled size "<<mkpts0_sampled.rows<<" "<<mkpts0_sampled.cols<<" "<<mkpts0_sampled.channels()<<std::endl;
    // std::cout<<"feats0_sampled  size "<<feats0_sampled .rows<<" "<<feats0_sampled.cols<<" "<<feats0_sampled.channels()<<std::endl;

    // cv::Mat cossim12_sampled = feats0_sampled * feats1_sampled.t();
	// cv::Mat cossim21_sampled = cossim12_sampled.t();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"reshape duration: "<<duration.count() /1000.0<<" seconds"<<std::endl;
	start = std::chrono::high_resolution_clock::now();
    cv::Mat cossim12 = feats0_reshaped * feats1_reshaped.t();
    //cv::Mat cossim21 = feats1_reshaped * feats0_reshaped.t();
    cv::Mat cossim21 = cossim12.t();
	end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"cossim duration: "<<duration.count() /1000.0<<" seconds"<<std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<int> match12(feats0_reshaped.rows, -1);
    for (int i = 0; i < cossim12.rows; i++){
        auto *row = cossim12.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for(int j = 0; j < cossim12.cols; j++){
            if(row[j] > maxScore){
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match12[i] = maxIdx;
    }

    std::vector<int> match21(feats1_reshaped.rows, -1);
    for (int i = 0; i < cossim21.rows; i++){
        auto *row = cossim21.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for(int j = 0; j < cossim21.cols; j++){
            if(row[j] > maxScore){
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match21[i] = maxIdx;
    }
	end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"match duration: "<<duration.count()/1000.0<<" seconds"<<std::endl;

    //todo: DMatch
    matches.clear();
    for(int i = 0; i < feats0_reshaped.rows; i++){
        int j = match12[i];
        if (match21[j] == i && cossim12.at<float>(i,j) > minScore){
            matches.emplace_back(i, j, cossim12.at<float>(i, j));
        }
    }

    // points
    for (int i = 0; i < mkpts0_reshaped.rows; i++){
        points1.push_back(cv::Point2f(mkpts0_reshaped.at<float>(i, 0), mkpts0_reshaped.at<float>(i, 1)));
        points2.push_back(cv::Point2f(mkpts1_reshaped.at<float>(i, 0), mkpts1_reshaped.at<float>(i, 1)));
    }

    return 0;
}

int XFeat::match_reshaped(const cv::Mat& mkpts0, const cv::Mat& feats0, const cv::Mat& mkpts1, const cv::Mat& feats1, std::vector<cv::DMatch>& matches, float minScore,
                    std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2){
    // std::vector<cv::Point2f> points_1;
    // std::vector<cv::Point2f> points_2;
     auto start = std::chrono::high_resolution_clock::now();
    cv::Mat mkpts0_reshaped = mkpts0.clone();
    cv::Mat mkpts1_reshaped = mkpts1.clone();
    cv::Mat feats0_reshaped = feats0.clone();
    cv::Mat feats1_reshaped = feats1.clone();
    // std::cout<<"mkpts0_reshaped size "<<mkpts0_reshaped.rows<<" "<<mkpts0_reshaped.cols<<" "<<mkpts0_reshaped.channels()<<std::endl;
    // std::cout<<"feats0_reshaped size "<<feats0_reshaped.rows<<" "<<feats0_reshaped.cols<<" "<<feats0_reshaped.channels()<<std::endl;
    
	// cv::Mat mkpts0_sampled = sampleMat(mkpts0_reshaped, 3, 1);
	// cv::Mat feats0_sampled = sampleMat( feats0_reshaped, 3, 1);
	// cv::Mat mkpts1_sampled = sampleMat(mkpts1_reshaped, 3, 1);
	// cv::Mat feats1_sampled = sampleMat( feats1_reshaped, 3, 1);
	// std::cout<<"mkpts0_sampled size "<<mkpts0_sampled.rows<<" "<<mkpts0_sampled.cols<<" "<<mkpts0_sampled.channels()<<std::endl;
    // std::cout<<"feats0_sampled  size "<<feats0_sampled .rows<<" "<<feats0_sampled.cols<<" "<<feats0_sampled.channels()<<std::endl;

    // cv::Mat cossim12_sampled = feats0_sampled * feats1_sampled.t();
	// cv::Mat cossim21_sampled = cossim12_sampled.t();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"reshape duration: "<<duration.count() /1000.0<<" seconds"<<std::endl;
	start = std::chrono::high_resolution_clock::now();
    cv::Mat cossim12 = feats0_reshaped * feats1_reshaped.t();
    //cv::Mat cossim21 = feats1_reshaped * feats0_reshaped.t();
    cv::Mat cossim21 = cossim12.t();
	end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"cossim duration: "<<duration.count() /1000.0<<" seconds"<<std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<int> match12(feats0_reshaped.rows, -1);
    for (int i = 0; i < cossim12.rows; i++){
        auto *row = cossim12.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for(int j = 0; j < cossim12.cols; j++){
            if(row[j] > maxScore){
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match12[i] = maxIdx;
    }

    std::vector<int> match21(feats1_reshaped.rows, -1);
    for (int i = 0; i < cossim21.rows; i++){
        auto *row = cossim21.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for(int j = 0; j < cossim21.cols; j++){
            if(row[j] > maxScore){
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match21[i] = maxIdx;
    }
	end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"match duration: "<<duration.count()/1000.0<<" seconds"<<std::endl;

    //todo: DMatch
    matches.clear();
    for(int i = 0; i < feats0_reshaped.rows; i++){
        int j = match12[i];
        if (match21[j] == i && cossim12.at<float>(i,j) > minScore){
            matches.emplace_back(i, j, cossim12.at<float>(i, j));
        }
    }

    // points
    for (int i = 0; i < mkpts0_reshaped.rows; i++){
        points1.push_back(cv::Point2f(mkpts0_reshaped.at<float>(i, 0), mkpts0_reshaped.at<float>(i, 1)));
        points2.push_back(cv::Point2f(mkpts1_reshaped.at<float>(i, 0), mkpts1_reshaped.at<float>(i, 1)));
    }

    return 0;
}

int XFeat::match_self(const cv::Mat& mkpts0, const cv::Mat& feats0, const cv::Mat& mkpts1, const cv::Mat& feats1, std::vector<cv::DMatch>& matches, float minScore,
                    std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2){
    // std::vector<cv::Point2f> points_1;
    // std::vector<cv::Point2f> points_2;
     auto start = std::chrono::high_resolution_clock::now();
    cv::Mat mkpts0_reshaped = mkpts0.reshape(1, mkpts0.size[1]);
    cv::Mat mkpts1_reshaped = mkpts1.reshape(1, mkpts1.size[1]);
    cv::Mat feats0_reshaped = feats0.reshape(1, feats0.size[1]);
    cv::Mat feats1_reshaped = feats1.reshape(1, feats1.size[1]);
    std::cout<<"mkpts0_reshaped size "<<mkpts0_reshaped.rows<<" "<<mkpts0_reshaped.cols<<" "<<mkpts0_reshaped.channels()<<std::endl;
    std::cout<<"feats0_reshaped size "<<feats0_reshaped.rows<<" "<<feats0_reshaped.cols<<" "<<feats0_reshaped.channels()<<std::endl;
    
	// cv::Mat mkpts0_sampled = sampleMat(mkpts0_reshaped, 3, 1);
	// cv::Mat feats0_sampled = sampleMat( feats0_reshaped, 3, 1);
	// cv::Mat mkpts1_sampled = sampleMat(mkpts1_reshaped, 3, 1);
	// cv::Mat feats1_sampled = sampleMat( feats1_reshaped, 3, 1);
	// std::cout<<"mkpts0_sampled size "<<mkpts0_sampled.rows<<" "<<mkpts0_sampled.cols<<" "<<mkpts0_sampled.channels()<<std::endl;
    // std::cout<<"feats0_sampled  size "<<feats0_sampled .rows<<" "<<feats0_sampled.cols<<" "<<feats0_sampled.channels()<<std::endl;

    // cv::Mat cossim12_sampled = feats0_sampled * feats1_sampled.t();
	// cv::Mat cossim21_sampled = cossim12_sampled.t();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"reshape duration: "<<duration.count() /1000.0<<" seconds"<<std::endl;
	start = std::chrono::high_resolution_clock::now();
    cv::Mat cossim12 = feats0_reshaped * feats1_reshaped.t();
    //cv::Mat cossim21 = feats1_reshaped * feats0_reshaped.t();
    cv::Mat cossim21 = cossim12.t();
	end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"cossim duration: "<<duration.count() /1000.0<<" seconds"<<std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<int> match12(feats0_reshaped.rows, -1);
    for (int i = 0; i < cossim12.rows; i++){
        auto *row = cossim12.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for(int j = 0; j < cossim12.cols; j++){
			if(i==j)
			   continue;
            if(row[j] > maxScore){
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match12[i] = maxIdx;
    }

    std::vector<int> match21(feats1_reshaped.rows, -1);
    for (int i = 0; i < cossim21.rows; i++){
        auto *row = cossim21.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for(int j = 0; j < cossim21.cols; j++){
			if(i==j)
			   continue;
            if(row[j] > maxScore){
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match21[i] = maxIdx;
    }
	end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout<<"match duration: "<<duration.count()/1000.0<<" seconds"<<std::endl;

    //todo: DMatch
    matches.clear();
    for(int i = 0; i < feats0_reshaped.rows; i++){
        int j = match12[i];
        if (match21[j] == i && cossim12.at<float>(i,j) > minScore){
            matches.emplace_back(i, j, cossim12.at<float>(i, j));
        }
    }

    // points
    for (int i = 0; i < mkpts0_reshaped.rows; i++){
        points1.push_back(cv::Point2f(mkpts0_reshaped.at<float>(i, 0), mkpts0_reshaped.at<float>(i, 1)));
        points2.push_back(cv::Point2f(mkpts1_reshaped.at<float>(i, 0), mkpts1_reshaped.at<float>(i, 1)));
    }

    return 0;
}

int XFeat::matchlightglue(const cv::Mat& mkpts0, const cv::Mat& feats0, const cv::Mat& sc0, const cv::Mat& mkpts1, const cv::Mat& feats1, cv::Mat& matches, cv::Mat& batch_indexes){
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	int64_t mkpts0_size = mkpts0.rows * mkpts0.cols * mkpts0.channels();
	std::vector<int64_t> mkpts0_dims = { mkpts0.rows, mkpts0.cols, mkpts0.channels()}; 
	Ort::Value mkpts0_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)(mkpts0.data), mkpts0_size, mkpts0_dims.data(), mkpts0_dims.size());

	int64_t feats0_size = feats0.rows * feats0.cols * feats0.channels();
	std::vector<int64_t> feats0_dims = { feats0.rows, feats0.cols, feats0.channels() }; 
	Ort::Value feats0_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)(feats0.data), feats0_size, feats0_dims.data(), feats0_dims.size());

	int64_t sc0_size = sc0.rows * sc0.cols;
	std::vector<int64_t> sc0_dims = { sc0.rows, sc0.cols }; // 1x4800
	Ort::Value sc0_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)(sc0.data), sc0_size, sc0_dims.data(), sc0_dims.size());

	int64_t mkpts1_size = mkpts1.rows * mkpts1.cols * mkpts1.channels();
	std::vector<int64_t> mkpts1_dims = { mkpts1.rows, mkpts1.cols, mkpts1.channels() }; 
	Ort::Value mkpts1_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)(mkpts1.data), mkpts1_size, mkpts1_dims.data(), mkpts1_dims.size());

	int64_t feats1_size = feats1.rows * feats1.cols * feats1.channels();
	std::vector<int64_t> feats1_dims = { feats1.rows, feats1.cols, feats1.channels() }; 
	Ort::Value feats1_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)(feats1.data), feats1_size, feats1_dims.data(), feats1_dims.size());

	// Create input tensors
	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(std::move(mkpts0_tensor));
	input_tensors.push_back(std::move(feats0_tensor));
	input_tensors.push_back(std::move(sc0_tensor));
	input_tensors.push_back(std::move(mkpts1_tensor));
	input_tensors.push_back(std::move(feats1_tensor));

	// Run session
	auto output_tensors =
		matchingSession_.Run(Ort::RunOptions{ nullptr }, matchingInputNames.data(),
			input_tensors.data(), matchingInputNames.size(), matchingOutputNames.data(), matchingOutputNames.size());
	assert(output_tensors.size() == xfeatOutputNames.size() && output_tensors.front().IsTensor());
	
	// Get outputs
	auto matchesShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	int dim1 = static_cast<int>(matchesShape[0]); // num
	int dim2 = static_cast<int>(matchesShape[1]); // 4
	// To cv::Mat
	float* matchesDataPtr = output_tensors[0].GetTensorMutableData<float>();
	matches = cv::Mat(dim1, dim2, CV_32F, matchesDataPtr).clone();

	auto batch_indexesShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
	dim1 = static_cast<int>(batch_indexesShape[0]); // num

	float* batch_indexesDataPtr = output_tensors[0].GetTensorMutableData<float>();
	batch_indexes = cv::Mat(dim1, 1, CV_32F, batch_indexesDataPtr).clone();

	return 0;
}