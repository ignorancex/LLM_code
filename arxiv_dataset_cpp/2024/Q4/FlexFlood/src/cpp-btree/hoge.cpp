#include <bits/stdc++.h>
#include "btree_set.h"
using namespace std;
using namespace btree;
int main(void) {
    btree_set<int> st;
    for(int i = 0; i < 10; i++) {
        st.insert(i);
    }
    cout << st.size() << '\n';
}