// These are the important learning that  i have made through mistakes
// 1. Whenever you are using Disjoint set union try to call the parent for each element once so that the actual parent of each node get set perfectly.
// 2. In case of fenwick try always using 1 indexed array with it as it can create problem in case of zero
// 3. In game theory question try to look at the problem in the form of a tree which is similiar to the three max min tree  which we have seen in AIES course work.
// 4. While attempting oa or contest on leet code try to use your own min and max function as they are only capable of dealing with integers not with long long
// 5. So whenver you declare a global variable let say count and try to use it in other functions then you will encounter an error like count is ambigious the main reason for that is under the namespace standard count is also present as function in header file which is causing ambiguity to resolve this issue do not use names which can be present in header files
// 6. Did you know about the counting sort algorithm which can perform sorting in O(n) complexity if all number in the range 1e7
// 7. See you need to just hash the given array and iterate over it and place the place in sorted order on the basis of the occurrence of the variables
// 8. just like upper_bound and lower_bound function we have an equal_range function which returns the lowest index and highest index for the given value
// in the same approach that lower_bound and upperbound use
// 9. 
// ⇒ Whenever we have to deal with range things which comes to mind is 
// Hash array
// Fenwick Tree
// Segment Tree

// ⇒ Now see Hash array is capable of producing the result in O(1) time for any range with O(n) precomputation.
// But fails when it comes to update the element in between querying

// ⇒ Here comes fenwick tree
// Fenwick tree is capable of producing the result for a range in O(logn) by precomputation of O(nlogn) initially 
// Also when it comes for updation[point update]  it can update in O(logn) time and produce results
// But fails when we have to deal with range queries

// ⇒ Also both of them are useless when you have to determine result of function where 
// f(l to r) != f(r) - f(l-1)    like minimum in a range and maximum in a range

// ⇒ But what if we have to deal with those function then we can use segment trees

// ⇒ Here comes the Segment Tree which with the precomputation of O(nlogn) time is capable of answering range queries in O(logn) time

// When it comes to Point update it is again take only O(logn) for updation
// When it comes to Range update it take only O(logn) time again with the concept of lazy propagation and 

// ⇒ Again there is a point let say we have function of type f(l to r) != f(r) - f(l-1) and query are static that there is no concept of changing the queries then we can use 

// ⇒ Sparse Table
// The are very similiar to binary lifting concept in binary tree where we construct a precompute array called lookup array of size n*logn such that if we have lookup[i][j] → then it means start from index i to 2^j -1 what is the result 

// To query in sparse table we have to first calcuate the size of range which needed to be calculated then we can divide the range in to two parts which is needt to be calculated let say we have [L,R] then it can be ([L,L+ (1<<j)-1] =lookup[L][j]) + ([R-(1<<j)+1],R]=lookup[R-(1<<j)+1][j] remember they have overlapping portion also which is [R-(1<<j)+1,L+(1<<j)-1]




#include<bits/stdc++.h>
using namespace std;
int main(){
    vector<int>v={1,2,3,3,3,3,3,3,3,4,4,4,4,5,6};
    auto it=equal_range(v.begin(),v.end(),3); 
    // [output will be two pointer one pointing at first occurrence another at last occurrence +1 ]
    cout<<int(it.first-v.begin())<<"   "<<int(it.second-v.begin())<<endl;
    it=equal_range(v.begin(),v.end(),4);
    cout<<int(it.first-v.begin())<<"   "<<int(it.second-v.begin())<<endl;
    return 0;
}


// 9. bitset are the data structure which can be used in case of 1e8 elements,and you can either represent these number in form of binary number , or these data structures are used when you need to do optimizations like in dp [related to memory optimization]

// bitset<SIZE> bt(string("111001010"))
// 10. Similarly we have priority queue
// priority_queue<int>pq;



// There are Policy based data structure which are supported by gnu compiler the normal data structure that we use belongs to STL

#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;

typedef tree<int,null_type,less<int>,rb_tree_tag,
tree_order_statistics_node_update> indexed_set;