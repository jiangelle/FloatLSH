#include <time.h>
#include <iostream>
#include <flann\algorithms\lsh_index.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <algorithm>
using namespace std;
using namespace flann;

#define LSH_FLOAT

#ifdef LSH_FLOAT
typedef L2_Simple<float> Distance;
#else
typedef Hamming<unsigned char> Distance;
#endif
typedef Distance::ResultType DistanceType;

void printIndexMatrix(flann::Matrix<size_t> matrix) {
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			printf("%d%s", matrix[i][j], j == matrix.cols-1 ? "\n" : ",");
		}
	}
}

int elapsedMilliseconds(clock_t startTime) {
	return 1000.0*(clock() - startTime) / CLOCKS_PER_SEC;
}

void printLSHResult(int k_nn_, const flann::Matrix<DistanceType>& gt_dists) {
	printf("lsh result\n");
	vector<float> minDistanceVector;
	vector<float> maxDistanceVector;
	vector<float> meanDistanceVector;
	vector<float> stdVarianceDistanceVector;
	for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
		minDistanceVector.push_back(FLT_MAX);
		maxDistanceVector.push_back(-FLT_MAX);
		meanDistanceVector.push_back(0);
		stdVarianceDistanceVector.push_back(0);
	}
	for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
		for (size_t row = 0; row < gt_dists.rows; ++row) {
			float distance = sqrt((float)gt_dists[row][neighbor_index]);
			minDistanceVector[neighbor_index] = min(minDistanceVector[neighbor_index], distance);
			maxDistanceVector[neighbor_index] = max(maxDistanceVector[neighbor_index], distance);
			meanDistanceVector[neighbor_index] += distance;
		}
		meanDistanceVector[neighbor_index] /= gt_dists.rows;
	}
	for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
		for (size_t row = 0; row < gt_dists.rows; ++row) {
			float distance = sqrt((float)gt_dists[row][neighbor_index]);
			float diff = distance - meanDistanceVector[neighbor_index];
			stdVarianceDistanceVector[neighbor_index] += diff*diff;
		}
		stdVarianceDistanceVector[neighbor_index] /= gt_dists.rows - 1;
		stdVarianceDistanceVector[neighbor_index] = sqrt(stdVarianceDistanceVector[neighbor_index]);
	}

	for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
		printf("neighbor_index=%d, min=%f, max=%f, mean=%f, stdVar=%f\n", neighbor_index, minDistanceVector[neighbor_index], maxDistanceVector[neighbor_index], meanDistanceVector[neighbor_index], stdVarianceDistanceVector[neighbor_index]);
	}
}

void printGroundTruthResult(int k_nn_, const flann::Matrix<Distance::ElementType>& query, const flann::Matrix<Distance::ElementType>& data) {
	printf("ground truth result\n");
	vector<vector<float>> groundTruthDistances;
	Distance distance_functor;
	for (int query_row = 0; query_row < query.rows; query_row++) {
		groundTruthDistances.push_back(vector<float>());
		vector<float> dataDistances;
		float* query_feature = new float[query.cols];
		for (size_t i = 0; i < query.cols; ++i) {
			query_feature[i] = query[query_row][i];
		}
		for (int i = 0; i < data.rows; i++) {
			float* data_feature = new float[query.cols];
			for (size_t j = 0; j < query.cols; ++j) {
				data_feature[j] = data[i][j];
			}
			float distance = sqrt(distance_functor(query_feature, data_feature, query.cols));
			dataDistances.push_back(distance);
			delete[] data_feature;
		}
		delete[] query_feature;
		sort(dataDistances.begin(), dataDistances.end());
		for (size_t i = 0; i < k_nn_; ++i) {
			groundTruthDistances[query_row].push_back(dataDistances[i]);
		}
	}
	vector<float> minDisVector;
	vector<float> maxDisVector;
	vector<float> meanDisVector;
	vector<float> stdVarDistanceVector;
	for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
		minDisVector.push_back(FLT_MAX);
		maxDisVector.push_back(-FLT_MAX);
		meanDisVector.push_back(0);
		stdVarDistanceVector.push_back(0);
	}
	for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
		for (size_t query_row = 0; query_row < query.rows; ++query_row) {
			float distance = groundTruthDistances[query_row][neighbor_index];
			minDisVector[neighbor_index] = min(minDisVector[neighbor_index], distance);
			maxDisVector[neighbor_index] = max(maxDisVector[neighbor_index], distance);
			meanDisVector[neighbor_index] += distance;
		}
		meanDisVector[neighbor_index] /= query.rows;
	}
	for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
		for (size_t query_row = 0; query_row < query.rows; ++query_row) {
			float distance = groundTruthDistances[query_row][neighbor_index];
			float diff = distance - meanDisVector[neighbor_index];
			stdVarDistanceVector[neighbor_index] += diff*diff;
		}
		stdVarDistanceVector[neighbor_index] /= query.rows - 1;
		stdVarDistanceVector[neighbor_index] = sqrt(stdVarDistanceVector[neighbor_index]);
	}

	for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
		printf("neighbor_index=%d, min=%f, max=%f, mean=%f, stdVar=%f\n", neighbor_index, minDisVector[neighbor_index], maxDisVector[neighbor_index], meanDisVector[neighbor_index], stdVarDistanceVector[neighbor_index]);
	}
}

int main(int argc, char** argv)
{
	flann::Matrix<Distance::ElementType> data;
	flann::Matrix<Distance::ElementType> query;
	flann::Matrix<size_t> gt_indices;
	flann::Matrix<DistanceType> gt_dists;
	unsigned int k_nn_ = 3;
	clock_t startTime = clock();
	printf("neighbor count=%d\n", k_nn_);
	printf("Reading test data...");
	fflush(stdout);
	flann::load_from_file(data, "brief100K.h5", "dataset");
	flann::load_from_file(query, "brief100K.h5", "query");
	printf("read data done, %d ms\n", elapsedMilliseconds(startTime));
	printf("data rows=%d, cols=%d\n", data.rows, data.cols);
	printf("query rows=%d, cols=%d\n", query.rows, query.cols);
	flann::Index<Distance> index(data, flann::LshIndexParams());
	startTime = clock();
	index.buildIndex();
	printf("build index done, %d ms\n", elapsedMilliseconds(startTime));

	printf("Searching KNN ...\n");
	gt_indices = flann::Matrix<size_t>(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	gt_dists = flann::Matrix<DistanceType>(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
	startTime = clock();
	index.knnSearch(query, gt_indices, gt_dists, k_nn_, flann::SearchParams(-1));
	printf("search knn done, %d ms\n", elapsedMilliseconds(startTime));
	//printIndexMatrix(gt_indices);
	printGroundTruthResult(k_nn_, query, data);
	printLSHResult(k_nn_, gt_dists);

	system("pause");
}
