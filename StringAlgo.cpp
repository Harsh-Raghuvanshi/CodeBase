#include <bits/stdc++.h>
using namespace std;

// Trie Code for numberic it can be converted to deal with number
// write -> Node* links[2]; -> denoting 0 and 1
struct Node
{
    Node *links[26];
    int CntEndWith = 0;
    int CntPrefix = 0;

    bool isExist(char ch) { return (links[ch - 'a'] != NULL); }
    Node *get(char ch) { return links[ch - 'a']; }
    void put(char ch, Node *temp) { links[ch - 'a'] = temp; }
    void increaseEnd() { CntEndWith++; }
    void increasePrefix() { CntPrefix++; }
    void deleteEnd() { CntEndWith--; }
    void reducePrefix() { CntPrefix--; }
    int getEnd() { return CntEndWith; }
    int getPrefix() { return CntPrefix; }
};
class Trie
{
private:
    Node *root;

public:
    Trie()
    {
        root = new Node();
    }
    void insertWord(string s)
    {
        int n = s.size();
        Node *node = root;
        for (int i = 0; i < n; i++)
        {
            if (!node->isExist(s[i]))
            {
                node->put(s[i], new Node());
            }
            node = node->get(s[i]);
            node->increasePrefix();
        }
        node->increaseEnd();
    }
    int CntWordEqualTo(string &word)
    {
        Node *node = root;
        for (int i = 0; i < int(word.size()); i++)
        {
            if (node->isExist(word[i]))
                node = node->get(word[i]);
            else
                return 0;
        }
        return node->getEnd();
    }
    int CntPrefixes(string &word)
    {
        Node *node = root;
        for (int i = 0; i < int(word.size()); i++)
        {
            if (node->isExist(word[i]))
                node = node->get(word[i]);
            else
                return 0;
        }
        return node->getPrefix();
    }
    void Erase(string &word)
    {
        Node *node = root;
        for (int i = 0; i < int(word.size()); i++)
        {
            if (node->isExist(word[i]))
            {
                node = node->get(word[i]);
                node->reducePrefix();
            }
            else
                return;
        }
        node->deleteEnd();
    }
};

// KMP algorithm all the occurences of pattern you will get in ans array
// just call like vector<int>occurences=KMPSearch(pattern,text);
// pattern->to be found in text
void computeLPSArray(string pat, int M, vector<int> &lps)
{
    int len = 0;
    lps[0] = 0;
    int i = 1;
    while (i < M)
    {
        if (pat[i] == pat[len])
        {
            len++;
            lps[i] = len;
            i++;
        }
        else
        {
            if (len != 0)
            {
                len = lps[len - 1];
            }
            else
            {
                lps[i] = 0;
                i++;
            }
        }
    }
}
vector<int> KMPSearch(string pat, string txt)
{
    int M = pat.size();
    int N = txt.size();
    vector<int> lps(M, 0), ans;
    computeLPSArray(pat, M, lps);
    int i = 0;
    int j = 0;
    while ((N - i) >= (M - j))
    {
        if (pat[j] == txt[i])
        {
            j++;
            i++;
        }
        if (j == M)
        {
            ans.push_back(i - j);
            j = lps[j - 1];
        }
        else if (i < N && pat[j] != txt[i])
        {
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }
    }
    return ans;
}

// Z-function this is alternative to kmp or know as extended kmp in this approach basically for a
// given string s we create and z array where ith index in z array i.e. z[i]=> represent the maximum length
// of a substring which is starting at the ith index and it is also the prefix of given string

// if      s="a  a  b  c  a  b  c  c  a  a  b"
// then    z= 11 |1 |0 |0 |1 |0 |0 |0 |3 |1 |0

vector<int> z_function(string s)
{
    int n = s.size();
    vector<int> z(n, 0);
    z[0] = n;
    int l = 0, r = 0;
    for (int i = 1; i < n; i++)
    {
        if (i < r)
        {
            z[i] = min(r - i, z[i - l]);
        }
        while (i + z[i] < n && s[z[i]] == s[i + z[i]])
        {
            z[i]++;
        }
        if (i + z[i] > r)
        {
            l = i;
            r = i + z[i];
        }
    }
    return z;
}

struct Manacher {
    vector<int> p;

    // Manacher's Algorithm to calculate the longest palindromic substring lengths
    void run_manacher(const string &s) {
        int n = s.length();
        p.assign(n, 0);
        int l = 0, r = 0;

        for (int i = 0; i < n; i++) {
            // Mirror position of i around center l
            if (i < r) 
                p[i] = min(r - i, p[l + r - i]);

            // Expand around center i
            while (i - p[i] - 1 >= 0 && i + p[i] + 1 < n && s[i - p[i] - 1] == s[i + p[i] + 1])
                p[i]++;
            
            // Update the rightmost palindrome's boundary
            if (i + p[i] > r) {
                l = i;
                r = i + p[i];
            }
        }
    }

    // Preprocesses the input string to insert '#' between characters
    void build(const string &s) {
        string t = "#";
        for (char ch : s) {
            t.push_back(ch);
            t.push_back('#');
        }
        run_manacher(t);
    }

    // Returns the length of the longest palindrome centered at 'center'
    int getLongest(int center, bool odd) {
        int pos = 2 * center + 1 + (!odd);
        if (pos < 0 || pos >= p.size()) 
            return 0;
        return p[pos];
    }

    // Checks if the substring s[l...r] is a palindrome
    bool checkPalin(int l, int r) {
        int len = r - l + 1;
        int center = (l + r) / 2;
        bool isOdd = (l & 1) == (r & 1);
        return len <= getLongest(center, isOdd);
    }
} m;

int main()
{
    return 0;
}