/*
	This program has been created by Stefan Zwaard, based on the work of Paul Baker. Current version: 0.5, 16-06-2020.
	The (Distributed) Model Training and Aggregation (DMTLA) program allows for training and aggregation of SVM (HOG based) and ERT models.
	The program is part of the EyeBlink Project, and works together with the Image processing and EAR calculation program to proces images using the newly created models.
	The program was also used as a proof of concept and for results gathering for the proposed Mean Weight Matrix Aggregation (MWMA) distributed training for L-SVM models, and Weighted Bin Aggregation (WBA) distributed training of ERT models, 
	in the paper: "Privacy-Preserving Algorithms for Object Detection & Localization Using Distributed Machine Learning".
	Running the resulting program requires opencv_world400.dll to be present in the same folder. In order to compile this code, ensure both the OpenCV and DLib libraries are linked.
	Always use the AVX extended instructions on Release x64 for best preformance of program
*/


#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/image_io.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <math.h>
#include <string>
#include <filesystem>

#include "Window.h"
#include "CERT.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_keypoint/draw_surf_points.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>

using namespace std;
using namespace cv;
using namespace dlib;

// Define scnaner type for SVM training
typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

// Here follows a few function definitions from DLib for SVM and ERT training
// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances(
	const std::vector<std::vector<full_object_detection> >& objects
);

void pick_best_window_size(
	const std::vector<std::vector<dlib::rectangle> >& boxes,
	unsigned long& width,
	unsigned long& height,
	const unsigned long target_size
)
/*!
	ensures
		- Finds the average aspect ratio of the elements of boxes and outputs a width
		  and height such that the aspect ratio is equal to the average and also the
		  area is equal to target_size.  That is, the following will be approximately true:
			- #width*#height == target_size
			- #width/#height == the average aspect ratio of the elements of boxes.
!*/
{
	// find the average width and height
	running_stats<double> avg_width, avg_height;
	for (unsigned long i = 0; i < boxes.size(); ++i)
	{
		for (unsigned long j = 0; j < boxes[i].size(); ++j)
		{
			avg_width.add(boxes[i][j].width());
			avg_height.add(boxes[i][j].height());
		}
	}

	// Adjusting the box size sp it is about target_pizels pixels in size 
	double size = avg_width.mean()*avg_height.mean();
	double scale = std::sqrt(target_size / size);

	width = (unsigned long)(avg_width.mean()*scale + 0.5);
	height = (unsigned long)(avg_height.mean()*scale + 0.5);
	// make sure the width and height never round to zero.
	if (width == 0)
		width = 1;
	if (height == 0)
		height = 1;
}



bool contains_any_boxes(
	const std::vector<std::vector<dlib::rectangle> >& boxes
)
{
	for (unsigned long i = 0; i < boxes.size(); ++i)
	{
		if (boxes[i].size() != 0)
			return true;
	}
	return false;
}

void throw_invalid_box_error_message(
	const std::string& dataset_filename,
	const std::vector<std::vector<dlib::rectangle> >& removed,
	const unsigned long target_size
)
{
	image_dataset_metadata::dataset data;
	load_image_dataset_metadata(data, dataset_filename);

	std::ostringstream sout;
	sout << "Error!  An impossible set of object boxes was given for training. ";
	sout << "All the boxes need to have a similar aspect ratio and also not be ";
	sout << "smaller than about " << target_size << " pixels in area. ";
	sout << "The following images contain invalid boxes:\n";
	std::ostringstream sout2;
	for (unsigned long i = 0; i < removed.size(); ++i)
	{
		if (removed[i].size() != 0)
		{
			const std::string imgname = data.images[i].filename;
			sout2 << "  " << imgname << "\n";
		}
	}
	throw dlib::error("\n" + wrap_string(sout.str()) + "\n" + sout2.str());
}

// ----------------------------------------------------------------------------------------

// General system variables, mostly used for the main GUI window
bool Start = 0;
int TrainSVMConfig = 0;
int AggregateSVMConfig = 0;
std::string SVMModelsInPath = "E:\\EyeBlink\\Models\\ToAggregate.svm";
std::string SVMModelOutPath = "E:\\EyeBlink\\Models\\NewModel.svm";
int TrainERTConfig = 0;
int AggregateERTConfig = 0;
std::string ERTModelsInPath = "E:\\EyeBlink\\Models\\ToAggregate.dat";
std::string ERTModelOutPath = "E:\\EyeBlink\\Models\\NewModel.dat";
std::string SVMTrainingDatasetPath = "E:\\EyeBlink\\Data\\SVMTrainingdata.xml";
std::string SVMTestDatasetPath = "E:\\EyeBlink\\Data\\SVMTestingdata.xml";
std::string ERTTrainingDatasetPath = "E:\\EyeBlink\\Data\\ERTTrainingdata.xml";
std::string ERTTestDatasetPath = "E:\\EyeBlink\\Data\\ERTTestingdata.xml";

// Variable decleration for SVM training settings, changed and set with the SVM training GUI
bool SVMDone = 0;
int SVMThreads = 6;
double C = 100.0;
double eps = 0.01; 
unsigned int num_folds = 3;
unsigned long target_size = 80 * 80;
unsigned long upsample_amount = 2;

// Variable decleration for ERT training settings, changed and set with the ERT training GUI
bool ERTDone = 0;
unsigned int ERTThreads = 6;
unsigned long oversampling_amount = 20; // Random deformations of input data to help overcome generalization
double nu = 0.1; // 0,1 for default, the lower the more generalized training, 1 for complete fitting
unsigned long tree_depth = 5; // Default 2 for normal, 8 for high accuracy, verry slow on higher values, also severly increases model size be extreemly carefull
unsigned int feature_pool_size = 400; // Number of pixels used for ERT feature processing in each cascade. 400 default
unsigned int num_test_splits = 20; // Default 50. Changes accuracy of training, but slows training speed
unsigned int cascade_depth = 10; // Number of iterations
unsigned long num_trees_per_cascade_level = 500; // Default is 500
double lambda = 0.1; 

int main(int argc, char** argv)
{
	try
	{
		// Showing main GUI
		win my_window(&Start, &TrainSVMConfig, &AggregateSVMConfig, &SVMModelsInPath, &SVMModelOutPath, &TrainERTConfig, &AggregateERTConfig, &ERTModelsInPath, &ERTModelOutPath, &SVMTrainingDatasetPath, &SVMTestDatasetPath, &ERTTrainingDatasetPath, &ERTTestDatasetPath);
		while (true)
		{
			while (Start == 0)
			{
				sleep(100); // Sleep while system waits for user to start program, Idle running can be implemented here if neeeded
			}
			cout << "Program has started" << endl;

			// If *.svm or *.dat is used to select more then one model of each type, for either multi model aggregation or testing, each model is seperated into its own file.
			std::vector<String> VectorSVMInputModelPaths;
			std::vector<String> VectorERTInputModelPaths; // Keep in mind, this path can also be used to load CERT models for testing only (option 3 in main GUI)
			std::vector<String> VectorCERTInputModelPaths; // Not used at the moment, could be used later if adding or removing Subdivisions from CERT model using other ERT models is needed. For now the ERT input is used.
			std::vector < object_detector<image_scanner_type>> SVMModels; // Vector for all loaded SVM models
			std::vector < shape_predictor> ERTModels; // Vector for all loaded ERT models
			std::vector < CERT > CERTModels; // Vector for all loaded CERT models

			if (AggregateSVMConfig == 1 || AggregateSVMConfig == 2) // Load SVM models from input path if needed
			{
				glob(SVMModelsInPath, VectorSVMInputModelPaths); // If a incursive function is wanted, where all subfolders of the target are also searched, add '1' to the arguments
				cout << "Number of .svm models detected on SVM model(s) input path:" << VectorSVMInputModelPaths.size() << endl;
				for (int i = 0; i < VectorSVMInputModelPaths.size(); i++)
				{
					ifstream fin(VectorSVMInputModelPaths[i], ios::binary);
					if (fin)
					{
						object_detector<image_scanner_type> TempDetector;
						cout << "Loaded .svm file for either multi model aggregation or test only testing, file:" << VectorSVMInputModelPaths[i] << endl;
						deserialize(TempDetector, fin);
						SVMModels.push_back(TempDetector);
					}
					else
						cout << "Error reading .svm file on SVM model(s) input path, file was:" << VectorSVMInputModelPaths[i] << endl;
				}
			}

			if (AggregateERTConfig == 1 || AggregateERTConfig == 2) // Load ERT models from input path if needed
			{
				glob(ERTModelsInPath, VectorERTInputModelPaths); // If a incursive function is wanted, where all subfolders of the target are also searched, add '1' to the arguments
				cout << "Number of .dat models detected on ERT model(s) input path:" << VectorERTInputModelPaths.size() << endl;
				for (int i = 0; i < VectorERTInputModelPaths.size(); i++)
				{
					ifstream fin(VectorERTInputModelPaths[i], ios::binary);
					if (fin)
					{
						shape_predictor TempDetector;
						cout << "Loaded .dat file for either Weighted Bin Aggregation or test only testing, file:" << VectorERTInputModelPaths[i] << endl;
						deserialize(TempDetector, fin);
						ERTModels.push_back(TempDetector);
					}
					else
						cout << "Error reading .dat file on ERT model(s) input path, file was:" << VectorERTInputModelPaths[i] << endl;
				}
			}

			if (AggregateERTConfig == 3) // Load CERT models from input path if needed (Uses ERT input path at the moment)
			{
				glob(ERTModelsInPath, VectorERTInputModelPaths); // If a incursive function is wanted, where all subfolders of the target are also searched, add '1' to the arguments
				cout << "Number of .CERT models detected on ERT model(s) input path:" << VectorERTInputModelPaths.size() << endl;
				for (int i = 0; i < VectorERTInputModelPaths.size(); i++)
				{
					ifstream fin(VectorERTInputModelPaths[i], ios::binary);
					if (fin)
					{
						CERT TempDetector;
						cout << "Loaded .CERT file for either WBA CERT model only testing, file:" << VectorERTInputModelPaths[i] << endl;
						TempDetector.deserialize(fin);
						CERTModels.push_back(TempDetector);
					}
					else
						cout << "Error reading .CERT file on CERT model(s) input path, file was:" << VectorERTInputModelPaths[i] << endl;
				}
			}

			if (TrainSVMConfig == 1) // Getting SVM training settings from user if needed
			{
				cout << "Showing SVM model training window now" << endl;
				SVMsettingsWin SVMWindow(&SVMDone, &SVMThreads, &C, &eps, &num_folds, &target_size, &upsample_amount);
				while (SVMDone == 0) 
				{
					sleep(100); // Wait while user finishes SVM training input
				}
				SVMDone = 0;
				// Callout settings back to user
				cout << "SVM model training configured, using the following settings:" << endl;
				cout << "threads:" << SVMThreads << endl;
				cout << "C:" << C << endl;
				cout << "eps:" << eps << endl;
				cout << "num_folds:" << num_folds << endl;
				cout << "target_size:" << target_size << endl;
				cout << "upsample_amount" << upsample_amount << endl;
			}

			if (TrainERTConfig == 1) // Getting ERT training settings from user if needed
			{
				cout << "Showing ERT model training window now" << endl;
				ERTsettingsWin ERTWindow(&ERTDone, &ERTThreads, &oversampling_amount, &nu, &tree_depth, &feature_pool_size, &num_test_splits, &cascade_depth, &num_trees_per_cascade_level, &lambda);
				while (ERTDone == 0)
				{
					sleep(100); // Wait while user finishes SVM training input
				}
				ERTDone = 0;
				// Callout settings back to user
				cout << "ERT model training configured, using the following settings:" << endl;
				cout << "threads:" << ERTThreads << endl;
				cout << "oversampling_amount:" << oversampling_amount << endl;
				cout << "nu:" << nu << endl;
				cout << "tree_depth:" << tree_depth << endl;
				cout << "feature_pool_size:" << feature_pool_size << endl;
				cout << "num_test_splits:" << num_test_splits << endl;
				cout << "cascade_depth:" << cascade_depth << endl;
				cout << "num_trees_per_cascade_level:" << num_trees_per_cascade_level << endl;
				cout << "lambda:" << lambda << endl;
			}

			// SVM Model training if needed (based on default DLib Training)
			object_detector<image_scanner_type> NewSVMDetector; // Placeholder for the new created SVM model
			if (TrainSVMConfig == 1) // Training new .svm model based on user settings
			{
				cout << "Starting preperations for SVM model training." << endl;
				dlib::array<array2d<unsigned char> > images;
				std::vector<std::vector<dlib::rectangle> > object_locations, ignore;

				cout << "Loading training image dataset from dataset .xml file " << SVMTrainingDatasetPath << endl;
				ignore = load_image_dataset(images, object_locations, SVMTrainingDatasetPath);
				cout << "Number of images loaded: " << images.size() << endl;

				// Check if there are more folds than there are images.  
				if (num_folds > images.size())
					num_folds = images.size();

				// Upsampling if needed
				for (unsigned long i = 0; i < upsample_amount; ++i)
					upsample_image_dataset<pyramid_down<2> >(images, object_locations, ignore);

				image_scanner_type scanner;
			
				unsigned long width, height;
				pick_best_window_size(object_locations, width, height, target_size);
				scanner.set_detection_window_size(width, height);
				structural_object_detection_trainer<image_scanner_type> trainer(scanner);

				trainer.set_num_threads(SVMThreads);
				trainer.be_verbose(); // This is always set, this gives a progress overview in the CLI during training, remove this is unwanted, or make it depending on user settings.
				trainer.set_c(C);
				trainer.set_epsilon(eps);

				// Making sure input trainingsdata has boxes usable by detector
				std::vector<std::vector<dlib::rectangle> > removed;
				removed = remove_unobtainable_rectangles(trainer, images, object_locations);
				// Error handeling for if a box does not match (this happens a lot if there is no upsampling)
				if (contains_any_boxes(removed))
				{
					unsigned long scale = upsample_amount + 1;
					scale = scale * scale;
					throw_invalid_box_error_message(SVMTrainingDatasetPath, removed, target_size / scale);
				}
				cout << "Preperations completed. Starting training on new SVM model" << endl;
				NewSVMDetector = trainer.train(images, object_locations, ignore);
				cout << "Training of new SVM  model has completed." << endl;
			}

			// ERT model training if needed (based on default DLib Training)
			shape_predictor NewERTPredictor; // Placeholder for the new ERT model created
			if (TrainERTConfig == 1) // training new .dat model based on user settings
			{
				cout << "Starting preperations for ERT model training." << endl;
				cout << "Loading training image dataset from dataset .xml file " << ERTTrainingDatasetPath << endl;

				dlib::array<array2d<unsigned char> > images_train;
				std::vector<std::vector<full_object_detection> > faces_train;
				load_image_dataset(images_train, faces_train, ERTTrainingDatasetPath);
				cout << "Number of images loaded: " << images_train.size() << endl;

				shape_predictor_trainer trainer;

				// Applying ERT training settings
				trainer.set_nu(nu);
				trainer.set_tree_depth(tree_depth);
				trainer.set_num_threads(ERTThreads);
				if (oversampling_amount > 0)
					trainer.set_oversampling_amount(oversampling_amount);
				trainer.set_cascade_depth(cascade_depth);
				trainer.set_feature_pool_size(feature_pool_size);
				trainer.set_num_test_splits(num_test_splits);
				trainer.set_num_trees_per_cascade_level(num_trees_per_cascade_level);
				trainer.set_lambda(lambda);

				trainer.be_verbose(); // This is always set, this gives a progress overview in the CLI during training, remove this is unwanted, or make it depending on user settings.

				cout << "Preperations completed. Starting training on new ERT model" << endl;

				NewERTPredictor = trainer.train(images_train, faces_train);
				cout << "Training of new ERT  model has completed." << endl;
				cout << endl << "Number of parts of ERT Model: " << NewERTPredictor.num_parts();
				cout << endl << "Number of feutures of of ERT Model" << NewERTPredictor.num_features() << endl;
			}

			// SVM Model aggregation if needed, using the Mean Weight Matrix Aggregation (MWMA) algorithm, see paper for detailed explenation
			if (AggregateSVMConfig == 1) // Aggregating new .svm model based on user settings, aggregates any trained .svm model as well
			{
				cout << "Starting SVM multi model aggregation process" << endl;
				cout << "Loaded " << SVMModels.size() << " SVM models from input path for aggregation proces" << endl;
				if (TrainSVMConfig == 1)
					cout << "Trained SVM model shall be included in aggregation proces" << endl;

				cout << "Starting SVM multi aggregation" << endl;

				matrix<double, 0, 1> TempW; // MWMA step 1: New temp matrix for aggregation
				TempW.set_size((SVMModels[0].get_w()).size());
				TempW = SVMModels[0].get_w(); // MWMA step 1: init of w using first model
				for (int i = 1; i < SVMModels.size(); i++)
				{
					TempW += SVMModels[i].get_w(); // MWMA step 2a: Adding all other matrix values
				}
				if (TrainSVMConfig == 1)
				{
					TempW += NewSVMDetector.get_w(); // MWMA step 2a: Inclusion of trained SVM model in aggregation
					TempW /= (SVMModels.size() + 1); // MWMA step 2b: averaging all matrix values including trained model
				}
				else
					TempW /= SVMModels.size(); // MWMA step 2b: averageing all matrix values

				NewSVMDetector = object_detector<image_scanner_type>(SVMModels[0].get_scanner(), SVMModels[0].get_overlap_tester(), TempW); // MWMA step 3: Feuture extraction transfer and saving
				cout << "SVM Multi aggregation completed" << endl;
			}

			// ERT Model aggregation if needed, using tthe Weighted Bin Aggregation (MWMA) algorithm, this combines several ERT models into a new Comnbined-ERT model (CERT), see paper for detailed explenation
			CERT NewCERT;
			if (AggregateERTConfig == 1) // Aggregating new ERT model based on user settings, aggregates trained .dat model as well
			{
				cout << "Starting ERT Weighted Bin Aggregation process" << endl;
				cout << "Loaded " << ERTModels.size() << " ERT models from input path for aggregation proces" << endl;
				if (TrainERTConfig == 1)
				{
					cout << "Trained ERT model shall be included in aggregation proces" << endl;
					ERTModels.push_back(NewERTPredictor);
				}

				// WBA Step 1: Forrest combination
				cout << "ERT WBA: Forrest combination" << endl;
				std::vector<dlib::shape_predictor> NewSubDivisions;
				for (int i = 0; i < ERTModels.size(); i++)
				{
					NewSubDivisions.push_back(ERTModels[i]);
				}

				// WBA Step 2: Devider calculation
				cout << "ERT WBA: Devider calculation" << endl;
				std::vector<double> NewDevider; // The devider represents how much the subdevision partakes in the calculation of the end result and should total 1
				std::vector<double> DeviderOffSet; // This vector contains all offset values in order to bias end outcome towards a certain model. This could be filled using user config later, but for now its hardcoded for euqal participation (by filling it with zero's only.
				double TotalDeviationValue = 0; // This number is needed to determin how the end result index of 1 is reached. If all moddels have equal weight, this is the number of models (to ensure mean average). If a model has more or less weight, then this model should be changed.
				for (int i = 0; i < ERTModels.size(); i++) // Offset calculation
				{
					DeviderOffSet.push_back(1); // Adding ones first always, a 1 means 100% base of this model is used for final weight calculation, so place in vector is filled even if no offset is given, 1's only means all models partake equaly. 
					DeviderOffSet[i] += 0; // Add specific model offset here (later trough GUI settings, but now hardcoded). For 20% more bias towards a model, add 0.2 (this ofcourse also means that other models lose participation value towards this model. Therefore If all models get the same increase in bias, it has no effect; 2's for all modes results in same for 1's in all.
					TotalDeviationValue += DeviderOffSet[i]; // If a model has increased or decreased weight, the toal deviation should always be changed, to ensure the later total result index always equals 1. 
				}
				for (int i = 0; i < ERTModels.size(); i++) // Final devider calculation, by pushing back the final deviation values now instead of in the first loop, changes can be made depending on the TotalDeviationValue if needed.
				{
					double TempBinDeviation;
					TempBinDeviation = DeviderOffSet[i];
					NewDevider.push_back(TempBinDeviation);
				}

				// WBA Step 3: Weighted Bins Creation
				cout << "ERT WBA: Weighted bins creation" << endl;
				std::vector<std::pair<dlib::full_object_detection, double>> NewWeightedBins;
				for (int i = 0; i < ERTModels.size(); i++) // Creation of all bins, one bin for each subdevision, this is used to store sub results and used in combination with the corresponding devider for end result calculation
				{
					dlib::full_object_detection NewBin;
					std::pair<dlib::full_object_detection, double> NewWeightedBin(NewBin, NewDevider[i]);
					NewWeightedBins.push_back(NewWeightedBin);
				}
				
				// WBA step 4: Averaging of init shape model
				cout << "ERT WBA: Aggregating shapes"<< endl;
				matrix<float, 0, 1> NewShape; // New init Shape for averaging
				NewShape.set_size((ERTModels[0].get_shape()).size());
				NewShape = ERTModels[0].get_shape();
				for (int i = 1; i < ERTModels.size(); i++)
				{
					NewShape += ERTModels[i].get_shape();

				}
				NewShape /= ERTModels.size(); // Averageing all matrix values

				NewCERT = CERT(NewSubDivisions, TotalDeviationValue, NewWeightedBins, NewShape); // Creation of new combi ERT model

				cout << "ERT WBA completed" << endl;
			}

			// Model testing and saving
			if (AggregateSVMConfig == 1  || TrainSVMConfig == 1) // Saving and testing new .svm model if one has been created
			{
				cout << "Saving newly created SVM model to disk." << endl;
				serialize(SVMModelOutPath) << NewSVMDetector; // saving new SVM file to disk

				// Testing newly created SVM model
				dlib::array<array2d<unsigned char> > images;
				std::vector<std::vector<dlib::rectangle> > object_locations, ignore;
				cout << "Loading Test image dataset from dataset.xml file " << SVMTestDatasetPath << endl;
				ignore = load_image_dataset(images, object_locations, SVMTestDatasetPath);
				cout << "Number of images loaded: " << images.size() << endl;
				cout << "Testing new SVM model on specified test dataset." << endl;
				cout << "Accuracy of SVM model (precision,recall,AP): " << test_object_detection_function(NewSVMDetector, images, object_locations, ignore) << endl;
			}

			if (AggregateSVMConfig == 2) // Testing selected input .svm models if test only mode is used
			{
				dlib::array<array2d<unsigned char> > images;
				std::vector<std::vector<dlib::rectangle> > object_locations, ignore;
				cout << "Loading SVM Test image dataset from dataset.xml file " << SVMTestDatasetPath << endl;
				ignore = load_image_dataset(images, object_locations, SVMTestDatasetPath);
				cout << "Number of images loaded: " << images.size() << endl;

				cout << "Testing:" << SVMModels.size() << " SVM model(s) on specified test dataset." << endl;
				for (int i = 0; i < SVMModels.size(); i++)
				{
					cout << "Testing SVM model:" << VectorSVMInputModelPaths[i] << endl;
					cout << "Accuracy of SVM model  (precision,recall,AP): " << test_object_detection_function(SVMModels[i], images, object_locations, ignore) << endl;
				}
				cout << "Testing of SVM models completed." << endl;
			}

			if (TrainERTConfig == 1) // Saving and testing new .dat model if one has been trained
			{
				cout << "Saving newly trained ERT model to disk." << endl;
				serialize(ERTModelOutPath) << NewERTPredictor; // saving new .dat file to disk

				cout << "Loading ERT Test image dataset from dataset.xml file " << ERTTestDatasetPath << endl;
				dlib::array<array2d<unsigned char> > images_test;
				std::vector<std::vector<full_object_detection> > faces_test;
				load_image_dataset(images_test, faces_test, ERTTestDatasetPath);
				cout << "Number of images loaded: " << images_test.size() << endl;

				cout << "Mean testing error of new ERT model: " <<
					test_shape_predictor(NewERTPredictor, images_test, faces_test, get_interocular_distances(faces_test)) << endl;  // Test ERT model using internal test software
			}

			if (AggregateERTConfig == 1) // Saving and testing new .CERT model if one has been created
			{
				cout << "Saving newly aggregated CERT model to disk." << endl;
				ofstream fout(ERTModelOutPath, ios::binary);
				if (fout)
				{
					NewCERT.serialize(fout);
				}
				else
					cout << "Error saving CERT model. Model output path was:" << ERTModelOutPath << endl;

				cout << "Loading ERT Test image dataset from dataset.xml file " << ERTTestDatasetPath << endl;
				dlib::array<array2d<unsigned char> > images_test;
				std::vector<std::vector<full_object_detection> > faces_test;
				load_image_dataset(images_test, faces_test, ERTTestDatasetPath);
				cout << "Number of images loaded: " << images_test.size() << endl;

				const std::vector<std::vector<double> > scales = get_interocular_distances(faces_test);
				running_stats<double> rs;
				for (unsigned long i = 0; i < faces_test.size(); ++i)
				{
					for (unsigned long j = 0; j < faces_test[i].size(); ++j)
					{
						const double scale = scales.size() == 0 ? 1 : scales[i][j]; // Scale needed for testing
					    full_object_detection Shape = NewCERT.PredictFinalShape(images_test[i], faces_test[i][j].get_rect()); // This provides a full_object_detection from a CERT model, use this function if a CERT model is used where normaly a ERT model is used.  
						for (unsigned long k = 0; k < Shape.num_parts(); k++) // The new CERT model has no internal test function like the default ERT code from DLib, therefore the Mean Error Rate is calculated here below.
						{
							double score = length(Shape.part(k) - faces_test[i][j].part(k)) / scale; 
							rs.add(score);
						}
					}
				}
				cout << "Mean testing error of new ERT model: " << rs.mean() << endl;
					
			}

			if (AggregateERTConfig == 2) // Testing selected input .dat models if test only mode is used. This tests ERT models, not the Combined-ERT models, can be used to see most suitable models for aggregation
			{
				cout << "Loading ERT Test image dataset from dataset.xml file " << ERTTestDatasetPath << endl;
				dlib::array<array2d<unsigned char> > images_test;
				std::vector<std::vector<full_object_detection> > faces_test;
				load_image_dataset(images_test, faces_test, ERTTestDatasetPath);
				cout << "Number of images loaded: " << images_test.size() << endl;

				cout << "Testing:" << ERTModels.size() << " ERT model(s) on specified test dataset." << endl;

				for (int i = 0; i < ERTModels.size(); i++) 
				{
					cout << "Testing ERT model:" << VectorERTInputModelPaths[i] << endl;
					cout << "Mean testing error of ERT model: " << test_shape_predictor(ERTModels[i], images_test, faces_test, get_interocular_distances(faces_test)) << endl; // Test ERT model using internal test software
				}

			}

			if (AggregateERTConfig == 3) // Testing selected input CERT .dat models if test only mode is used
			{
				cout << "Loading ERT Test image dataset from dataset.xml file " << ERTTestDatasetPath << endl;
				dlib::array<array2d<unsigned char> > images_test;
				std::vector<std::vector<full_object_detection> > faces_test;
				load_image_dataset(images_test, faces_test, ERTTestDatasetPath);
				cout << "Number of images loaded: " << images_test.size() << endl;

				cout << "Testing:" << CERTModels.size() << " CERT model(s) on specified test dataset." << endl;

				for (int m = 0; m < CERTModels.size(); m++)
				{
					cout << "Testing CERT model:" << VectorERTInputModelPaths[m] << endl;
					const std::vector<std::vector<double> > scales = get_interocular_distances(faces_test);
					running_stats<double> rs;
					for (unsigned long i = 0; i < faces_test.size(); ++i)
					{
						for (unsigned long j = 0; j < faces_test[i].size(); ++j)
						{
							const double scale = scales.size() == 0 ? 1 : scales[i][j]; // Scale needed for testing
							full_object_detection Shape = CERTModels[m].PredictFinalShape(images_test[i], faces_test[i][j].get_rect()); // This provides a full_object_detection from a CERT model, use this function if a CERT model is used where normaly a ERT model is used.  
							for (unsigned long k = 0; k < Shape.num_parts(); k++) // The new CERT model has no internal test function like the default ERT code from DLib, therefore the Mean Error Rate is calculated here below.
							{
								double score = length(Shape.part(k) - faces_test[i][j].part(k)) / scale;
								rs.add(score);
							}
						}
					}
					cout << "Mean testing error of CERT model: " << rs.mean() << endl;
				}
			}

			Start = 0; // System is done and shall await next usage
			cout << "Program has completed task, a new task can now be started using the start button, please change settings where needed." << endl;
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
		cout << "Restart of program needed, close window or terminate program from CLI." << endl;
	}
}


// Implementaiton of remaining DLib functions for testing and training of ERT models, needed to calculate the scale
// ----------------------------------------------------------------------------------------


double interocular_distance(
	const full_object_detection& det
)
{
	dlib::vector<double, 2> l, r;
	double cnt = 0;
	// Find the center of the left eye by averaging the points around 
	// the eye.
	for (unsigned long i = 36; i <= 41; ++i)
	{
		l += det.part(i);
		++cnt;
	}
	l /= cnt;

	// Find the center of the right eye by averaging the points around 
	// the eye.
	cnt = 0;
	for (unsigned long i = 42; i <= 47; ++i)
	{
		r += det.part(i);
		++cnt;
	}
	r /= cnt;

	// Now return the distance between the centers of the eyes
	return length(l - r);
}

std::vector<std::vector<double> > get_interocular_distances(
	const std::vector<std::vector<full_object_detection> >& objects
)
{
	std::vector<std::vector<double> > temp(objects.size());
	for (unsigned long i = 0; i < objects.size(); ++i)
	{
		for (unsigned long j = 0; j < objects[i].size(); ++j)
		{
			temp[i].push_back(interocular_distance(objects[i][j]));
		}
	}
	return temp;
}

