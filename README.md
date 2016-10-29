Locality sensitive hashing for float type is modified under the Locality sensitive hashing for integer.
And the mainly changes that has been made are in the lsh_index.h and the lsh_table.h.
In lsh_table.h, the add() function which is used to add points into table has been modified from unsigned char into float,a well as the getBucketFromKey() function.
The getKey() function has used the newly hashing function for the float type.The function randomFloat() is used to compute b in the report.The function gaussrand() 
is used to compute the random vector a.The function computeBucketSize() is used to compute the bandWidth W.
In lsh_index.h, the only change has been made is the getNeighbors() function.The getNeighbors() function for float is wroten.The difference from the origin is the Distance
type has been modified from Hamming Distance to L2_simple(in NN_index.h),and the modified version has ignored the xor_masks to get the neghboring buckets of the keyBucket.
In the main.cpp ,the printGroundTruthResult() function is writen to output the result of brute_force searching,the printLSHResult() function is writen to compare the 
search result from the ground truth.
