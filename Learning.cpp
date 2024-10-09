// These are the important learning that  i have made through mistakes
// 1. Whenever you are using Disjoint set union try to call the parent for each element once so that the actual parent of each node get set perfectly.
// 2. In case of fenwick try always using 1 indexed array with it as it can create problem in case of zero
// 3. In game theory question try to look at the problem in the form of a tree which is similiar to the three max min tree  which we have seen in AIES course work.
// 4. While attempting oa or contest on leet code try to use your own min and max function as they are only capable of dealing with integers not with long long
// 5. So whenver you declare a global variable let say count and try to use it in other functions then you will encounter an error like count is ambigious the main reason for that is under the namespace standard count is also present as function in header file which is causing ambiguity to resolve this issue do not use names which can be present in header files
// 6. Did you know about the counting sort algorithm which can perform sorting in O(n) complexity if all number in the range 1e7
// 7. See you need to just hash the given array and iterate over it and place the place in sorted order on the basis of the occurrence of the variables
// 8. just like upper_bount and lower_bound function we have an equal_range function which returns the lowest index and highest index for the given value
// in the same approach that lower_bound and upperbound use

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