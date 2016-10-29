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

class LshIndex_Brief100K
{
public:
	flann::Matrix<Distance::ElementType> data;
	flann::Matrix<Distance::ElementType> query;
	typedef Distance::ResultType DistanceType;
	flann::Matrix<size_t> gt_indices;
	flann::Matrix<DistanceType> gt_dists;
	unsigned int k_nn_;

	void run()
	{
		k_nn_ = 3;
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
		
		printf("Searching KNN for ground truth...\n");
		gt_indices = flann::Matrix<size_t>(new size_t[query.rows * k_nn_], query.rows, k_nn_);
		gt_dists = flann::Matrix<DistanceType>(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
		startTime = clock();
		index.knnSearch(query, gt_indices, gt_dists, k_nn_, flann::SearchParams(-1));
		printf("search knn done, %d ms\n", elapsedMilliseconds(startTime));
		printIndexMatrix(gt_indices);

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
				float distance = (float)gt_dists[row][neighbor_index];
				minDistanceVector[neighbor_index] = min(minDistanceVector[neighbor_index], distance);
				maxDistanceVector[neighbor_index] = max(maxDistanceVector[neighbor_index], distance);
				meanDistanceVector[neighbor_index] += distance;
			}
			meanDistanceVector[neighbor_index] /= gt_dists.rows;
		}
		for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
			for (size_t row = 0; row < gt_dists.rows; ++row) {
				float distance = (float)gt_dists[row][neighbor_index];
				float diff = distance - meanDistanceVector[neighbor_index];
				stdVarianceDistanceVector[neighbor_index] += diff*diff;
			}
			stdVarianceDistanceVector[neighbor_index] /= gt_dists.rows-1;
			stdVarianceDistanceVector[neighbor_index] = sqrt(stdVarianceDistanceVector[neighbor_index]);
		}

		for (size_t neighbor_index = 0; neighbor_index < k_nn_; ++neighbor_index) {
			printf("neighbor_index=%d, min=%f, max=%f, mean=%f, stdVar=%f\n", neighbor_index, minDistanceVector[neighbor_index], maxDistanceVector[neighbor_index], meanDistanceVector[neighbor_index], stdVarianceDistanceVector[neighbor_index]);
		}

		system("pause");
		delete[] data.ptr();
		delete[] query.ptr();
		delete[] gt_indices.ptr();
		delete[] gt_dists.ptr();
	}
};

int main(int argc, char** argv)
{
	LshIndex_Brief100K abc;
	abc.run();
}
