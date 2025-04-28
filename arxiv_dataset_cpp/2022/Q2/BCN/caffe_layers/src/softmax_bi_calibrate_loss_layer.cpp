#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/customlayers/softmax_bi_calibrate_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

//  Implemetation of the loss for Bi-Calibration Networks
//  Last modified 6/18/2021

namespace caffe {

	// **********************
	// The input:
	// bottom[0]: query_pred
	// bottom[1]: query_label
	// bottom[2]: text_pred
	// bottom[3]: text_label
	// **********************
	// The output: 
	// top[0]: loss 
	// **********************


	/// Forward the query and text prediction through softmax 
	/// to obtain the query_prob and text_prob
	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		LayerParameter softmax_param(this->layer_param_);
		softmax_param.set_type("Softmax");
		softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
		softmax_bottom_vec_.clear();
		softmax_bottom_vec_.push_back(bottom[0]);
		softmax_top_vec_.clear();
		softmax_top_vec_.push_back(&prob_);
		softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

		LayerParameter softmax_param_text(this->layer_param_);
		softmax_param_text.set_type("Softmax");
		softmax_text_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param_text);
		softmax_bottom_text_vec_.clear();
		softmax_bottom_text_vec_.push_back(bottom[2]);
		softmax_top_text_vec_.clear();
		softmax_top_text_vec_.push_back(&prob_text_);
		softmax_text_layer_->SetUp(softmax_bottom_text_vec_, softmax_top_text_vec_);

		has_ignore_label_ =
			this->layer_param_.loss_param().has_ignore_label();
		if (has_ignore_label_) {
			ignore_label_ = this->layer_param_.loss_param().ignore_label();
		}
		if (!this->layer_param_.loss_param().has_normalization() &&
			this->layer_param_.loss_param().has_normalize()) {
			normalization_ = this->layer_param_.loss_param().normalize() ?
			LossParameter_NormalizationMode_VALID :
												  LossParameter_NormalizationMode_BATCH_SIZE;
		}
		else {
			normalization_ = this->layer_param_.loss_param().normalization();
		}
		build_query_text_map();

	}


	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
		softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
		outer_num_ = bottom[0]->count(0, softmax_axis_); // batch_size
		inner_num_ = bottom[0]->count(softmax_axis_ + 1); // 1
		CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
			<< "Number of labels must match number of predictions; "
			<< "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
			<< "label count (number of labels) must be N*H*W, "
			<< "with integer values in {0, 1, ..., C-1}.";
		if (top.size() >= 2) {
			// softmax output
			top[1]->ReshapeLike(*bottom[0]);
		}
	}


	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::build_query_text_map(){
		const string& query_text_map = this->layer_param_.softmax_bi_calibrate_loss_param().source();
		q_threshold = this->layer_param_.softmax_bi_calibrate_loss_param().t2q_threshold();
		t_threshold = this->layer_param_.softmax_bi_calibrate_loss_param().q2t_threshold();
		alpha = this->layer_param_.softmax_bi_calibrate_loss_param().alpha();
		num_train_ = this->layer_param_.softmax_bi_calibrate_loss_param().num_train();
		std::ifstream infile_label(query_text_map.c_str());
		int count = 0;
		string label_tmp;
		while (infile_label >> label_tmp){
			int query_label = std::stoi(label_tmp);
			query_text_cluster_map.insert(std::make_pair(query_label, count));
			text_cluster_query_map[count] = query_label;
			count += 1;
		}
		for (int i = 0; i < num_train_; ++i){
			vector<Dtype> text_label_tmp(count, Dtype(0.0));
			history_text_label_all.push_back(text_label_tmp);
		}
		epoch_index_ = 0;
		train_index_ = 0;
	}


	template <typename Dtype>
	Dtype SoftmaxBiCalibrateLossLayer<Dtype>::get_normalizer(
		LossParameter_NormalizationMode normalization_mode, int valid_count) {
		Dtype normalizer;
		switch (normalization_mode) {
		case LossParameter_NormalizationMode_FULL:
			normalizer = Dtype(outer_num_ * inner_num_);
			break;
		case LossParameter_NormalizationMode_VALID:
			if (valid_count == -1) {
				normalizer = Dtype(outer_num_ * inner_num_);
			}
			else {
				normalizer = Dtype(valid_count);
			}
			break;
		case LossParameter_NormalizationMode_BATCH_SIZE:
			normalizer = Dtype(outer_num_);
			break;
		case LossParameter_NormalizationMode_NONE:
			normalizer = Dtype(1);
			break;
		default:
			LOG(FATAL) << "Unknown normalization mode: "
				<< LossParameter_NormalizationMode_Name(normalization_mode);
		}
		// Some users will have no labels for some examples in order to 'turn off' a
		// particular loss in a multi-task setup. The max prevents NaNs in that case.
		return std::max(Dtype(1.0), normalizer);
	}


	/// text_prob -> t2q correction (bottom-up aggregation)
	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::get_t2q_correction(
		const Dtype* text_prob_data, int query_dim){
		for (int query_label = 0; query_label < query_dim; query_label++){
			Dtype t2q_correction = 0;
			auto iter_query_text_prob = query_text_cluster_map.equal_range(query_label);
			if (iter_query_text_prob.first != std::end(query_text_cluster_map)){
				for (auto iter = iter_query_text_prob.first; iter != iter_query_text_prob.second; ++iter)
					t2q_correction += text_prob_data[iter->second];
			}
			t2q_correction_vec_.push_back(t2q_correction);
		}
	}


	/// query_prob -> q2t correction (top-down decomposition)
	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::get_q2t_correction(
		const Dtype* query_prob_data, int text_dim){
		for (int text_label = 0; text_label < text_dim; text_label++){
			int query_label = text_cluster_query_map[text_label];
			int text_prototype_num = 0;
			auto iter_query_text_prob = query_text_cluster_map.equal_range(query_label);
			for (auto iter = iter_query_text_prob.first; iter != iter_query_text_prob.second; ++iter) { text_prototype_num++; }
			Dtype q2t_correction = query_prob_data[query_label] / text_prototype_num;
			q2t_correction_vec_.push_back(q2t_correction);
		}
	}


	template <typename Dtype>
	Dtype SoftmaxBiCalibrateLossLayer<Dtype>::compute_l2_norm(
		vector<Dtype> vector_a, vector<Dtype> vector_b){
		Dtype diff_norm = 0;
		for (int i = 0; i < vector_a.size(); ++i){
			diff_norm += (vector_a[i] - vector_b[i])*(vector_a[i] - vector_b[i]);
		}
		diff_norm = sqrt(diff_norm);
		return diff_norm;
	}


	/// compute the l2 norm of differences between correction and confidences
	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::get_differences(
		const Dtype* query_prob_data, int query_label,
		const Dtype* text_prob_data, const Dtype* text_label, int query_dim, int text_dim){
		// compute the differences between correction and confidences
		vector<Dtype> query_confidence(query_dim, Dtype(0.0));
		query_confidence[query_label] = query_prob_data[query_label];
		vector<Dtype> text_confidence(text_dim, Dtype(0.0));
		for (int i = 0; i < text_dim; ++i) text_confidence[i] = text_prob_data[i] * text_label[i];
		t2q_diff = compute_l2_norm(query_confidence, t2q_correction_vec_);
		q2t_diff = compute_l2_norm(text_confidence, q2t_correction_vec_);
	}


	/// Calibration Selection Scheme to generate refined supervision
	/// t2q correction + query confidence -> refined query label
	/// q2t correction + text_confidence -> refined text label
	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::get_refined_query_text_label(
		const Dtype* query_prob_data, int query_label, const Dtype* text_prob_data, const Dtype* text_label, int query_dim, int text_dim){
		// compute the confidence
		vector<Dtype> query_confidence(query_dim, Dtype(0.0));
		query_confidence[query_label] = query_prob_data[query_label];
		vector<Dtype> text_confidence(text_dim, Dtype(0.0));
		for (int i = 0; i < text_dim; ++i) text_confidence[i] = text_prob_data[i] * text_label[i];
		// should use text to refine query
		if (t2q_diff > q_threshold && q2t_diff < t_threshold){
			Dtype norm_sum = 0;
			for (int i = 0; i < t2q_correction_vec_.size(); ++i) refined_query_label.push_back(query_confidence[i] + query_prob_data[i]);
			for (int i = 0; i < refined_query_label.size(); ++i) norm_sum += refined_query_label[i];
			for (int i = 0; i < refined_query_label.size(); ++i) refined_query_label[i] /= norm_sum;
			for (int i = 0; i < q2t_correction_vec_.size(); ++i) refined_text_label.push_back(text_label[i]);
		}
		// should use query to refine text
		else if (t2q_diff < q_threshold && q2t_diff > t_threshold){
			Dtype norm_sum = 0;
			for (int i = 0; i < q2t_correction_vec_.size(); ++i) refined_text_label.push_back(text_confidence[i] + text_prob_data[i]);
			for (int i = 0; i < refined_text_label.size(); ++i) norm_sum += refined_text_label[i];
			for (int i = 0; i < refined_text_label.size(); ++i) refined_text_label[i] /= norm_sum;
			// moving-average update strategy 
			// the epoch_index_ must be more than zero, since history_text_label_all has store all of the history text label
			if (epoch_index_ > 0){
				for (int i = 0; i < refined_text_label.size(); ++i){
					refined_text_label[i] = (1 - alpha) * refined_text_label[i] + alpha * history_text_label_all[train_index_][i];
				}
			}
			for (int i = 0; i < t2q_correction_vec_.size(); ++i) refined_query_label.push_back(Dtype(0.0));
			refined_query_label[query_label] = Dtype(1.0);
		}
		// use original query and text supervision for optimization
		else{
			for (int i = 0; i < q2t_correction_vec_.size(); ++i) refined_text_label.push_back(text_label[i]);
			for (int i = 0; i < t2q_correction_vec_.size(); ++i) refined_query_label.push_back(Dtype(0.0));
			refined_query_label[query_label] = Dtype(1.0);
		}
		// store history text label
		for (int i = 0; i < refined_text_label.size(); ++i) history_text_label_all[train_index_][i] = refined_text_label[i];
		train_index_++;
		if (train_index_ >= num_train_){
			train_index_ = 0;
			epoch_index_++;
		}
	}


	/// forward to obtain loss 
	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// The forward pass computes the softmax prob values.
		softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
		softmax_text_layer_->Forward(softmax_bottom_text_vec_, softmax_top_text_vec_);
		const Dtype* query_prob_data = prob_.cpu_data();
		const Dtype* text_prob_data = prob_text_.cpu_data();
		const Dtype* query_label = bottom[1]->cpu_data();
		const Dtype* text_label = bottom[3]->cpu_data();
		int dim = prob_.count() / outer_num_;
		int text_dim = prob_text_.count() / outer_num_;
		int count = 0;
		Dtype loss = 0;
		refined_query_label_all.clear();
		refined_text_label_all.clear();
		for (int i = 0; i < outer_num_; ++i) {
			int query_start_pos = i*dim;
			int text_start_pos = i*text_dim;
			const int query_label_value = static_cast<int>(query_label[i]);
			refined_query_label.clear();
			refined_text_label.clear();
			t2q_correction_vec_.clear();
			q2t_correction_vec_.clear();
			// compute t2q correction
			get_t2q_correction(text_prob_data + text_start_pos, dim);
			// compute q2t correction
			get_q2t_correction(query_prob_data + query_start_pos, text_dim);
			// obtain the l2 norm of the difference between correction and confidence
			get_differences(query_prob_data + query_start_pos, query_label_value,
				text_prob_data + text_start_pos, text_label + text_start_pos, dim, text_dim);
			// obtain the refined text and query supervision by Calibration Selection Scheme
			get_refined_query_text_label(query_prob_data + query_start_pos, query_label_value,
				text_prob_data + text_start_pos, text_label + text_start_pos, dim, text_dim);
			// store refined supervion in a batch
			refined_query_label_all.push_back(refined_query_label);
			refined_text_label_all.push_back(refined_text_label);
			for (int j = 0; j < inner_num_; j++) {
				// forward the query classification loss with refined query supervision
				for (int c = 0; c < dim; ++c){
					Dtype refined_value = refined_query_label[c];
					DCHECK_GE(refined_value, 0);
					DCHECK_LT(refined_value, prob_.shape(softmax_axis_));
					if (has_ignore_label_ && query_label_value == ignore_label_) { continue; }
					loss -= refined_value * log(std::max(query_prob_data[i*dim + c*inner_num_ + j], Dtype(FLT_MIN)));
				}
				// forward the text classification loss with refined text supervision
				for (int c = 0; c < text_dim; ++c){
					Dtype refined_value = refined_text_label[c];
					DCHECK_GE(refined_value, 0);
					DCHECK_LT(refined_value, prob_text_.shape(softmax_axis_));
					loss -= refined_value * log(std::max(text_prob_data[i*dim + c*inner_num_ + j], Dtype(FLT_MIN)));
				}
				++count;
			}
		}
		top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
		if (top.size() == 2) {
			top[1]->ShareData(prob_);
		}
	}


	/// backward to obtain gradients
	template <typename Dtype>
	void SoftmaxBiCalibrateLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1] || propagate_down[3]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs and text prototype label inputs.";
		}
		// compute the gradient for query prediction input
		if (propagate_down[0]) {
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* prob_data = prob_.cpu_data();
			caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
			int dim = prob_.count() / outer_num_;
			int count = 0;
			for (int i = 0; i < outer_num_; ++i) {
				for (int j = 0; j < inner_num_; ++j) {
					for (int qlabel = 0; qlabel < dim; qlabel++){
						if (has_ignore_label_ && qlabel == ignore_label_) {
							for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
								bottom_diff[i * dim + c * inner_num_ + j] = 0;
							}
						}
						else {
							Dtype refined_qlabel = refined_query_label_all[i][qlabel];
							bottom_diff[i * dim + qlabel * inner_num_ + j] += prob_.cpu_data()[i * dim + qlabel * inner_num_ + j] * refined_qlabel;
							bottom_diff[i * dim + qlabel * inner_num_ + j] -= refined_qlabel;
							++count;
						}
					}
				}
			}
			// Scale gradient
			Dtype loss_weight = top[0]->cpu_diff()[0] /
				get_normalizer(normalization_, count);
			caffe_scal(prob_.count(), loss_weight, bottom_diff);
		}
		// compute the gradient for text prediction input
		if (propagate_down[2]){
			Dtype* bottom_diff = bottom[2]->mutable_cpu_diff();
			const Dtype* prob_text_data = prob_text_.cpu_data();
			caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
			int text_dim = prob_text_.count() / outer_num_;
			int count = 0;
			for (int i = 0; i < outer_num_; ++i) {
				for (int j = 0; j < inner_num_; ++j) {
					for (int tlabel = 0; tlabel < text_dim; tlabel++){
						Dtype refined_tlabel = refined_text_label_all[i][tlabel];
						bottom_diff[i * text_dim + tlabel * inner_num_ + j] += prob_.cpu_data()[i * text_dim + tlabel * inner_num_ + j] * refined_tlabel;
						bottom_diff[i * text_dim + tlabel * inner_num_ + j] -= refined_tlabel;
						++count;
					}
				}
			}
			// Scale gradient
			Dtype loss_weight = top[0]->cpu_diff()[0] /
				get_normalizer(normalization_, count);
			caffe_scal(prob_text_.count(), loss_weight, bottom_diff);
		}
	}

	//#ifdef CPU_ONLY
	//STUB_GPU(SoftmaxBiCalibrateLossLayer);
	//#endif

	INSTANTIATE_CLASS(SoftmaxBiCalibrateLossLayer);
	REGISTER_LAYER_CLASS(SoftmaxBiCalibrateLoss);

}  // namespace caffe
