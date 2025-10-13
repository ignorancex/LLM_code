#include "onnx_helper.h"

bool initOrtSession(Ort::Env& env, Ort::Session& session, std::string& modelPath, const int& gpuId = 0)
{
	const ORTCHAR_T* ortModelPath = modelPath.c_str();

	bool sessionIsAvailable = false;

	if (sessionIsAvailable == false)
	{
		try //gpu
		{
			Ort::SessionOptions session_options;
			session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

			OrtCUDAProviderOptions cuda0ptions;
			cuda0ptions.device_id = gpuId;
			cuda0ptions.gpu_mem_limit = 4 << 30;

			session_options.AppendExecutionProvider_CUDA(cuda0ptions);

			session = Ort::Session(env, ortModelPath, session_options);
			
			sessionIsAvailable = true;
			std::cout << "Using accelerator: CUDA" << std::endl;
		}
		catch (Ort::Exception e)
		{
			// std::cout << "Exception code: " << e.GetOrtErrorCode() << ", exception: " << e.what() << std::endl;
			// std::cout << "Failed to init CUDA accelerator, Trying another accelerator..." << std::endl;
			sessionIsAvailable = false;
		}
		catch (...)
		{
			// std::cout << "Failed to init CUDA accelerator, Trying another accelerator..." << std::endl;
			sessionIsAvailable = false;
		}
	}
    
	if (sessionIsAvailable == false)
	{
		try //cpu
		{
			Ort::SessionOptions session_options;
			session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); //

			session = Ort::Session(env, ortModelPath, session_options);
			
			sessionIsAvailable = true;
			// std::cout << "Using accelerator: CPU" << std::endl;
		}
		catch (Ort::Exception e)
		{
			std::cout << "Exception code: " << e.GetOrtErrorCode() << ", exception: " << e.what() << std::endl;
			std::cout << "Failed to init CPU accelerator, Trying another accelerator..." << std::endl;
			sessionIsAvailable = false;
		}
		catch (...)
		{
			std::cout << "Failed to init CPU accelerator." << std::endl;
			sessionIsAvailable = false;
		}
	}

    //输出模型相关参数
	if (sessionIsAvailable == true)
	{
		Ort::AllocatorWithDefaultOptions allocator;
		// 获得输入节点的数量，当输入仅为RGB图像时，输入节点数量就为1
		size_t num_input_nodes = session.GetInputCount();

		// 获得输入节点的name，type和shape
		for (int i = 0; i < num_input_nodes; i++)
		{
			
			// Name
			std::string input_name = std::string(session.GetInputName(i, allocator));

			// std::cout << "Input " << i << ": " << input_name << ", shape: (";

			// Type
			Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

			ONNXTensorElementDataType type = tensor_info.GetElementType();

			// Shape
			std::vector<int64_t> input_node_dims = tensor_info.GetShape();

			for (int j = 0; j < input_node_dims.size(); j++) {
				// std::cout << input_node_dims[j];
				if (j == input_node_dims.size() - 1)
				{
					// std::cout << ")" << std::endl; 
				}
				else
				{
					// std::cout << ", ";
				}
			}
		}

		// 获得输出节点的数量，在这里xfeat输出为特征点，描述子，得分矩阵，输出节点数量为3
		size_t num_output_nodes = session.GetOutputCount();

		// 获得输出节点的name，type，shape
		for (int i = 0; i < num_output_nodes; i++) {
			// Name
			std::string output_name = std::string(session.GetOutputName(i, allocator));
			// std::cout << "Output " << i << ": " << output_name << ", shape: (";

			// type
			Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

			ONNXTensorElementDataType type = tensor_info.GetElementType();

			// shape
			std::vector<int64_t> output_node_dims = tensor_info.GetShape();
			for (int j = 0; j < output_node_dims.size(); j++) {
				// std::cout << output_node_dims[j];
				if (j == output_node_dims.size() - 1)
				{
					// std::cout << ")" << std::endl; 
				}
				else
				{
					// std::cout << ", ";
				}
			}
		}
		
	}
	else
	{
		std::cout << modelPath << " is invalid model." << std::endl;
	}

	return sessionIsAvailable;
}