#include <bits/stdc++.h>
using namespace std;
const int mod = 998244353;
const int Mod = 1e9 + 7;
#define ll long long
typedef vector<long long> vll;
typedef vector<int> vi;
#define py cout << "YES" << endl
#define pn cout << "NO" << endl
// #include <ext/pb_ds/assoc_container.hpp>
// using namespace __gnu_pbds;
// typedef tree<int,null_type,less<int>,rb_tree_tag,
// tree_order_statistics_node_update> indexed_set;

// to use this

template <class T>
int bitct(T a)
{
    int i = 0;
    while (a)
    {
        a /= 2;
        i++;
    }
    return i;
}

template <class T>
void printv(vector<T> a)
{
    for (T &el : a)
        cout << el << " ";
    cout << endl;
}

ll max(ll a, ll b)
{
    if (a > b)
        return a;
    return b;
}

ll min(ll a, ll b)
{
    if (a > b)
        return b;
    return a;
}

ll gcd(ll a, ll b)
{
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

struct custom_hash
{
    static uint64_t splitmix64(uint64_t x)
    {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const
    {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
// unordered_map<long long, int, custom_hash> safe_map;
// gp_hash_table<long long, int, custom_hash> safe_hash_table;

const int Mod = 1e9 + 7;
int bin_iter(int a, int b)
{
    int ans = 1;
    while (b)
    {
        if (b & 1)
            ans = (ans * 1LL * a) % Mod;
        a = (a * 1LL * a) % Mod;
        b >>= 1;
    }
    return ans;
}

// DSU Code
// Disjoint set union has a limitation that if you have joined two large groups then parent of all those element of large group doest no change completely in one go.
// So to avoid this scenario try to call parent function for each and every node again which will ensure that every child is correctly assigned with its respective parent group.
void make(int i, vector<int> &parent, vector<int> &sizes)
{
    parent[i] = i;
    sizes[i] = 1;
}
int findParent(int i, vector<int> &parent)
{
    if (i == parent[i])
        return i;
    return parent[i] = findParent(parent[i], parent);
}
void unions(int i, int j, vector<int> &parent, vector<int> &sizes)
{
    i = findParent(i, parent);
    j = findParent(j, parent);
    if (i == j)
        return;
    if (sizes[i] < sizes[j])
        swap(i, j);
    parent[j] = i;
    sizes[i] += sizes[j];
    sizes[j] = 0;
}

const int M = 1e7 + 2;
vector<bool> isprime(M, true);
vector<int> hp(M);
void sieve()
{
    isprime[0] = isprime[1] = false;
    for (int i = 2; i < M; i++)
    {
        if (isprime[i])
        {
            hp[i] = i;
            for (int j = 2 * i; j < M; j += i)
            {
                isprime[j] = false;
                hp[j] = i;
            }
        }
    }
}

// This function will return -1 if Kth ancestor do not exist
int getKthAncestor(int node, int k, vector<vector<int>> &dp)
{
    int ans = node;
    int j = 0;
    while (k)
    {
        if (ans == -1)
            break;
        if (k & 1)
        {
            ans = dp[j][ans];
        }
        k >>= 1;
        j++;
    }
    return ans;
}
void binary_lifting(int vertex, int parent, vector<vector<int>> &g, vector<vector<int>> &dp)
{
    dp[0][vertex] = parent;
    int temp_parent = parent;
    for (int i = 1; i < 20; i++)
    {
        if (temp_parent == -1)
            break;
        dp[i][vertex] = dp[i - 1][temp_parent];
        temp_parent = dp[i][vertex];
    }
    for (auto &child : g[vertex])
    {
        if (child == parent)
            continue;
        binary_lifting(child, vertex, g, dp);
    }
}
int findlca(int v, int u, int d, vector<vector<int>> &dp)
{
    if (u == v)
        return v;
    int lo = 0, hi = d, mid, anc, anc1;
    while (hi - lo > 1)
    {
        mid = (hi + lo) / 2;
        anc = getKthAncestor(v, mid, dp);
        anc1 = getKthAncestor(u, mid, dp);
        if (anc == anc1)
            hi = mid;
        else
            lo = mid + 1;
    }
    anc = getKthAncestor(v, lo, dp);
    anc1 = getKthAncestor(u, lo, dp);
    if (anc == anc1)
        return anc;
    return getKthAncestor(v, hi, dp);
}

// O(nlogn) ==> approach
int longestIncreasingSubsequence(vector<int> &arr, int n)
{
    vector<int> temp;
    temp.push_back(arr[0]);

    int len = 1;

    for (int i = 1; i < n; i++)
    {
        if (arr[i] > temp.back())
        {
            temp.push_back(arr[i]);
            len++;
        }
        else
        {
            int ind = lower_bound(temp.begin(), temp.end(), arr[i]) - temp.begin();
            temp[ind] = arr[i];
        }
    }

    return len;
}





// Rabin karp algorithm based on hashing of string and then matching the corresponding value
vector<int> rabin_karp(string const &s, string const &t)
{
    const int p = 31;
    const int m = 1e9 + 9;
    int S = s.size(), T = t.size();

    vector<long long> p_pow(max(S, T));
    p_pow[0] = 1;
    for (int i = 1; i < (int)p_pow.size(); i++)
        p_pow[i] = (p_pow[i - 1] * p) % m;

    vector<long long> h(T + 1, 0);
    for (int i = 0; i < T; i++)
        h[i + 1] = (h[i] + (t[i] - 'a' + 1) * p_pow[i]) % m;
    long long h_s = 0;
    for (int i = 0; i < S; i++)
        h_s = (h_s + (s[i] - 'a' + 1) * p_pow[i]) % m;

    vector<int> occurrences;
    for (int i = 0; i + S - 1 < T; i++)
    {
        long long cur_h = (h[i + S] + m - h[i]) % m;
        if (cur_h == h_s * p_pow[i] % m)
            occurrences.push_back(i);
    }
    return occurrences;
}



// fenwick tree code deals with updates log(n) time and provide prefix sum in also log(n) time.
// Remember while copying fenwick code indexing must start with ********* 1 only  **********
const int N = 1e6 + 4;
vector<int> fenwick(N, 0);
void update_fen(int i, int val)
{
    while (i < N)
    {
        fenwick[i] += val;
        i += (i & (-i));
    }
}
int sum_fen(int i)
{
    int sum = 0;
    while (i > 0)
    {
        sum += fenwick[i];
        i -= (i & (-i));
    }
    return sum;
}
// Binary lifting in fenwick trees this is for the purpose when we want to calculate the index where prefix sum is the lower bound of give value 'K'
int NN = 1e6 + 4;
// the size of array
// we can take it 24 also  as number of bits is not more than 24
int Fenwick_Lowbound(int k)
{
    int prevsum = 0, curr = 0;
    for (int i = log2(NN); i >= 0; i--)
    {
        if (fenwick[curr + (1 << i)] + prevsum < k)
        {
            curr = curr + (1 << i);
            prevsum += fenwick[curr];
        }
    }
    return curr + 1;
}

// Segment trees we will write for both point update and range update in range update we use lazy propagation so that we can save our time
// Also we are implementing segment tree for sum you can change it with you accordance
int N = 1e5 + 3;
vector<int> arr(N), seg(4 * N);
void build(int ind, int low, int high)
{
    if (low == high)
    {
        seg[ind] = arr[low];
        return;
    }
    int mid = (low + high) / 2;
    build(2 * ind + 1, low, mid);
    build(2 * ind + 2, mid + 1, high);
    seg[ind] += (seg[2 * ind + 1] + seg[2 * ind + 2]);
}
// low (0) and high(n-1) -> these are enpoints of whole array and it will vary by use where as
// l and r -> are the range whose result we have to compute and it will not vary
int query(int ind, int low, int high, int l, int r)
{
    if (low >= l and high <= r)
        return seg[ind];
    if (r < low || l > high)
        return 0;
    int mid = (low + high) / 2;
    int left = query(2 * ind + 1, low, mid, l, r);
    int right = query(2 * ind + 2, mid + 1, high, l, r);
    return left + right;
}
// now dealing with point update
// ind (0) , low (0) , high(n-1)  -> it will vary where node is the index and val is updated value.
void pointUpdate(int ind, int low, int high, int node, int val)
{
    if (low == high)
    {
        seg[ind] += val;
    }
    else
    {
        int mid = (low + high) / 2;
        if (node >= low and node <= mid)
        {
            pointUpdate(2 * ind + 1, low, mid, node, val);
        }
        else
            pointUpdate(2 * ind + 2, mid + 1, high, node, val);
        seg[ind] = (seg[2 * ind + 1] + seg[2 * ind + 2]);
    }
}
// now dealing with range update in case of it we will do lazy update
// for this we create a replica of Segment tree which will keep account of lazy update hence we call it lazy tree
vector<int> lazy(4 * N);
vector<int> seg(4 * N);

void rangeUpdate(int ind, int low, int high, int l, int r, int val)
{
    if (lazy[ind] != 0)
    {
        seg[ind] += (high - low + 1) * lazy[ind];
        
        
        if (low != high)
        {
            lazy[2 * ind + 1] += lazy[ind];  
            lazy[2 * ind + 2] += lazy[ind];  
        }
        
        lazy[ind] = 0;
    }

    if (r < low || l > high)
        return;

    if (low >= l && high <= r)
    {
        seg[ind] += (high - low + 1) * val;
        
        if (low != high)
        {
            lazy[2 * ind + 1] += val;  
            lazy[2 * ind + 2] += val;  
        }
        return;
    }

    int mid = (low + high) / 2;
    rangeUpdate(2 * ind + 1, low, mid, l, r, val);
    rangeUpdate(2 * ind + 2, mid + 1, high, l, r, val);

    seg[ind] = (seg[2 * ind + 1] + seg[2 * ind + 2]);
}


int queryLazy(int ind, int low, int high, int l, int r)
{
    if (lazy[ind] != 0)
    {
        seg[ind] += (high - low + 1) * lazy[ind];
        if (low != high)
        {
            lazy[2 * ind + 1] = lazy[ind];
            lazy[2 * ind + 2] = lazy[ind];
        }
        lazy[ind] = 0;
    }
    if (r < low || l > high || low > high)
        return 0;
    if (low >= l and high <= r)
    {
        return seg[ind];
    }
    int mid = (low + high) / 2;
    return queryLazy(2 * ind + 1, low, mid, l, r) + queryLazy(2 * ind + 2, mid + 1, high, l, r);
}

vector<int> Dijkstra(vector<pair<int, int>> g[], int source, int N)
{
    set<pair<int, int>> st;
    vector<bool> vis(N + 1, false);
    vector<int> dist(N + 1, int(1e9 + 7));
    st.insert({0, source});
    dist[source] = 0;

    while (st.size() > 0)
    {
        auto fr = (*(st.begin()));
        int pardist = fr.first, parnode = fr.second;
        st.erase(st.begin());

        if (vis[parnode])
            continue;
        vis[parnode] = true;

        for (auto &el : g[parnode])
        {
            int cnode = el.first, cdist = el.second;
            if (dist[cnode] > cdist + pardist)
            {
                dist[cnode] = cdist + pardist;
                st.insert({dist[cnode], cnode});
            }
        }
    }
    return dist;
}

void Floyd_warshall(vector<vector<int>> &dist, int N)
{
    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
}

vector<int> Topo_sort(vector<vector<int>> g, int N)
{
    vector<int> indegree(N, 0);
    vector<int> topo;
    for (int i = 0; i < N; i++)
    {
        for (int &el : g[i])
            indegree[el]++;
    }
    queue<int> q;
    for (int i = 0; i < N; i++)
    {
        if (indegree[i] == 0)
            q.push(i);
    }
    while (!q.empty())
    {
        int node = q.front();
        q.pop();
        topo.push_back(node);
        for (int &el : g[node])
        {
            indegree[el]--;
            if (indegree[el] == 0)
                q.push(el);
        }
    }
    return topo;
}

// To get Bridges in a graph using tarjan's algorithm
// just call getBridges function ans will give you with bridges maing thing is that the connections that are given in getBridges function just contain edges list so first we are preparing an adjacency matrix list.

int timer = 1;
void dfs(int vertex, int parent, vector<vector<int>> &g, vector<bool> &vis, vector<int> &tin, vector<int> &low, vector<vector<int>> &ans)
{
    vis[vertex] = true;
    tin[vertex] = low[vertex] = timer;
    timer++;
    for (auto child : g[vertex])
    {
        if (child == parent)
            continue;
        if (vis[child])
        {
            low[vertex] = min(low[vertex], tin[child]);
        }
        else
        {
            dfs(child, vertex, g, vis, tin, low, ans);
            low[vertex] = min(low[vertex], low[child]);
            if (low[child] > tin[vertex])
            {
                ans.push_back({vertex, child});
            }
        }
    }
}
vector<vector<int>> getBridges(int n, vector<vector<int>> &connections)
{
    vector<vector<int>> g(n);
    for (auto el : connections)
    {
        g[el[0]].push_back(el[1]);
        g[el[1]].push_back(el[0]);
    }
    vector<int> tin(n), low(n);
    vector<vector<int>> ans;
    vector<bool> vis(n, false);
    dfs(0, -1, g, vis, tin, low, ans);
    return ans;
}

// For finding articulation point in the graph articulation points are those points or nodes on whose removal the graph will get divided into two components
// See here in articulationPoints function we have already prepared adjacency list
int timer = 1;
void dfs(int node, int parent, vector<int> &vis, int tin[], int low[],
         vector<int> &mark, vector<int> adj[])
{
    vis[node] = 1;
    tin[node] = low[node] = timer;
    timer++;
    int child = 0;
    for (auto it : adj[node])
    {
        if (it == parent)
            continue;
        if (!vis[it])
        {
            dfs(it, node, vis, tin, low, mark, adj);
            low[node] = min(low[node], low[it]);
            if (low[it] >= tin[node] && parent != -1)
            {
                mark[node] = 1;
            }
            child++;
        }
        else
        {
            low[node] = min(low[node], tin[it]);
        }
    }
    if (child > 1 && parent == -1)
    {
        mark[node] = 1;
    }
}

vector<int> articulationPoints(int n, vector<int> adj[])
{
    vector<int> vis(n, 0);
    int tin[n];
    int low[n];
    vector<int> mark(n, 0);
    for (int i = 0; i < n; i++)
    {
        if (!vis[i])
        {
            dfs(i, -1, vis, tin, low, mark, adj);
        }
    }
    vector<int> ans;
    for (int i = 0; i < n; i++)
    {
        if (mark[i] == 1)
        {
            ans.push_back(i);
        }
    }
    if (ans.size() == 0)
        return {-1};
    return ans;
}

// DFS for Hierholzer's Algorithm for Directed graph

// This Algorithm is used to find eulerian path or circuit if exists in a directed graph
// The main logic in algo is that for any node you invoke a dfs function over its child and when dfs get stuck you start backtracking and pushing them into answer array,  and reach the node again ,  if the node still has some more child visit them end at last push the node also in answer array
// First create a adjacency list to represent the directed graph  also prepare netout map which states that how many net edges are going outward from a node
// pick the node whose netout is 1 because it will be the starting if no such node pick a node with an outgoing edge
// we can also convert dfsDirected function into finding the hamiltonian approach by adding a condition while entering the function which is if(st.find(s)!=st.end())return it means if we have intiated call for a string then we will not reach here first we have to go for another child [i.e do post order traversal]
bool isEulerianDirected(map<int, int> &netout, int &startNode)
{
    int ct = 0;
    for (auto &el : netout)
    {
        if (abs(el.second) > 1)
            return false;
        else if (abs(el.second) == 1)
            ct++;
        if (el.second == 1)
        {
            startNode = el.first;
        }
    }
    if (ct == 2)
    {
        cout << "It has only Eulerian PATH" << endl;
    }
    if (ct == 0)
    {
        cout << "It has Eulerian PATH and CIRCUIT" << endl;
    }
    return ct == 2 || ct == 0;
}
void dfsDirected(int vertex, vector<vector<int>> &g, vector<int> &ans)
{
    while (!g[vertex].empty())
    {
        int el = g[vertex].back();
        g[vertex].pop_back();
        dfsDirected(el, g, ans);
    }
    ans.push_back(vertex);
}

// DFS for Hierholzer's Algorithm for Undirected graph
void dfsUndirected(int vertex, vector<vector<int>> &graph, vector<int> &ans, map<pair<int, int>, bool> &usedEdges)
{
    while (!graph[vertex].empty())
    {
        int neighbor = graph[vertex].back();
        graph[vertex].pop_back();

        // Skip if the edge is already used
        if (usedEdges[{vertex, neighbor}] || usedEdges[{neighbor, vertex}])
            continue;

        // Mark the edge as used in both directions
        usedEdges[{vertex, neighbor}] = true;
        usedEdges[{neighbor, vertex}] = true;

        // Recur for the neighbor
        dfsUndirected(neighbor, graph, ans, usedEdges);
    }
    ans.push_back(vertex);
}

// Check if the graph has an Eulerian Path or Circuit
bool isEulerianUndirected(const vector<vector<int>> &graph, int &startNode, int &oddCount)
{
    int n = graph.size();
    oddCount = 0;
    startNode = -1;

    for (int i = 0; i < n; i++)
    {
        if (graph[i].size() % 2 != 0)
        {
            oddCount++;
            startNode = i; // Start from any odd degree node
        }
        else if (graph[i].size() > 0 && startNode == -1)
        {
            startNode = i; // If all degrees are even, start from any non-isolated node
        }
    }

    // Eulerian Circuit if all degrees are even
    // Eulerian Path if exactly 2 nodes have odd degree
    return (oddCount == 0 || oddCount == 2);
}

// I have to write backtrack algo for the finding hamiltonian path if it exists
// In hamiltonian circuit or path approach basically we have to visit all the nodes in manner such that no nodes will get visited twice
void findHamiltonian(vector<vector<int>> &g, vector<int> &path, int vertex, vector<bool> &vis, vector<int> &ans)
{
    vis[vertex] = true;
    path.push_back(vertex);
    if (all_of(vis.begin(), vis.end(), [](bool x)
               { return x == true; }))
    {
        ans = path;
    }
    for (auto &child : g[path.back()])
    {
        if (!vis[child])
            findHamiltonian(g, path, child, vis, ans);
    }
    vis[vertex] = false;
    path.pop_back();
}

// Legandre's theorem which is use to overcome the problem of :Given an integer n and a prime number p, the task is to find the largest x such that px (p raised to power x) divides n!.
//  Returns largest power of p that divides n!
int largestPower(int n, int p)
{
    int res = 0;

    // Calculate res = n/p + n/(p^2) + n/(p^3) + ....
    while (n > 0)
    {
        n /= p;
        res += n;
    }
    return res;
}

// Sparse Table
// Fills lookup array
// lookup[][] in bottom up manner. [ this is min lookup array]
void preprocess(vector<int> &arr, int n, vector<vector<int>> &lookup)
{
    // Initialize M for the
    // intervals with length 1
    for (int i = 0; i < n; i++)
        lookup[i][0] = i;

    // Compute values from smaller
    // to bigger intervals
    for (int j = 1; (1 << j) <= n; j++)
    {
        // Compute minimum value for
        // all intervals with size
        // 2^j
        for (int i = 0; (i + (1 << j) - 1) < n; i++)
        {
            // For arr[2][10], we
            // compare arr[lookup[0][3]]
            // and arr[lookup[3][3]]
            if (arr[lookup[i][j - 1]] < arr[lookup[i + (1 << (j - 1))][j - 1]])
                lookup[i][j] = lookup[i][j - 1];
            else
                lookup[i][j] = lookup[i + (1 << (j - 1))][j - 1];
        }
    }
}

// Returns minimum of arr[L..R]
int query(vector<int> &arr, int L, int R, vector<vector<int>> &lookup)
{
    // For [2,10], j = 3
    int j = (int)log2(R - L + 1);

    // For [2,10], we compare arr[lookup[0][3]] and
    // arr[lookup[3][3]],
    if (arr[lookup[L][j]] <= arr[lookup[R - (1 << j) + 1][j]])
        return arr[lookup[L][j]];

    else
        return arr[lookup[R - (1 << j) + 1][j]];
}


// This is merge sort but , it is also capable of calculating the number of inversion present in arr
// we know inversion is basically if i<j but arr[i]>arr[j]
// This will also mean it is giving the minimum number of ADJACENT swaps we need to sort the array completely
ll countInversions(vector<ll>&arr,ll l,ll r){
    if(l==r)return 0;
    ll mid=(l+r)>>1;
    ll x=countInversions(arr,l,mid);
    ll y=countInversions(arr,mid+1,r);
    ll inv=0;
    vector<ll>temp;
    ll i=l,j=mid+1;
    while(i<=mid && j<=r){
        if(arr[i]<arr[j]){
            temp.push_back(arr[i]);
            i++;
            inv+=(j-(mid+1));
        }else{
            temp.push_back(arr[j]);
            j++;
        }
    }
    while(i<=mid){
        inv+=(j-(mid+1));
        temp.push_back(arr[i]);
        i++;
    }
    while(j<=r){
        temp.push_back(arr[j]);
        j++;
    }
    ll ct=0;
    for(ll k=l;k<=r;k++){
        arr[k]=temp[ct];
        ct++;
    }
    return x+y+inv;
}


int main()
{
    return 0;
}

// CORE LEARNINGS :
// 1. Remeber whenever you are using Fenwick tree then you have to ensure that indexing begin with 1
// 2. if you want to fill an array with a praticular value then you can write like :
//     fill(v.begin(),v.end(),val);
// 3. Whenever you are using Disjoint Set Union the ensure that you call parent of every again before doing any
// computation as while joining two groups the parent may not get modified

// If you know manhattan distance between two point let say (x1,y1) and (x2,y2) form (x,y) then there is formula for calculating
// (x,y) as focus on matrix and draw simple lines between point these representing manhattan distance

#include <bits/stdc++.h>
#define ll long long
using namespace std;
#ifndef ONLINE_JUDGE
#define debug(x)       \
    cout << #x << " "; \
    _print(x);         \
    cout << endl;
#define sep() cout << "************************************" << endl;
#else
#define debug(x)
#define sep()
#endif
template <class T>
void _print(T var)
{
    cout << var << " ";
}
template <class T1, class T2>
void _print(pair<T1, T2> p)
{
    cout << "{" << p.first << "," << p.second << "}" << " ";
}
template <class T>
void _print(vector<T> v)
{
    cout << "[";
    for (T i : v)
        _print(i), _print(',');
    cout << "]\n";
}
template <class T>
void _print(set<T> st)
{
    cout << "[";
    for (T i : st)
        _print(i), _print(',');
    cout << "]\n";
}
template <class T1, class T2>
void _print(map<T1, T2> mp)
{
    cout << "[\n";
    for (auto i : mp)
    {
        _print("{"), _print(i.first), _print(":"), _print(i.second), _print("}");
        cout << endl;
    }
    cout << "]\n";
}

void solve()
{
    vector<int> v = {1, 2, 3};
    vector<pair<int, int>> vp = {{1, 1}, {2, 2}, {3, 3}};
    map<int, string> mp = {{1, "harsh"}, {2, "rohan"}, {3, "mridul"}};
    set<string> st = {"cow", "dog", "buffalo"};
    debug(v);
    sep();
    debug(mp);
    sep();
    debug(st);
    sep();
    debug(vp);
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    int t;
    cin >> t;
    while (t--)
        solve();
    return 0;
}