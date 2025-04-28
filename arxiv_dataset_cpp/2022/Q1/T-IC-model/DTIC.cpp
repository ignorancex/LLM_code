#include "DTIC.h"

typedef struct PARAMETER {
    int t1, t2;
    int nIters;
    unordered_set<int> * A;
    DiGraph * G;
    double spread;
} Parameter;

void * simulationThread(void * ptr) {
    Parameter * parameter = (Parameter *) ptr;

    int t1 = parameter->t1;
    int t2 = parameter->t2;
    int N = parameter->nIters;
    unordered_set<int> * A = parameter->A;
    DiGraph * G = parameter->G;

    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 mt_rand(seed);
    uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

    queue<int> Q;

    double avg = 0;

    for (int n = 0; n < N; n++) {

        unordered_set<int> visited, S;
        S.insert(A->begin(), A->end());

        for (int i = t1; i <= t2; i++) {

            for (int u : S) {
                visited.insert(u);
                Q.push(u);
            }

            while (!Q.empty()) {
                int u = Q.front();
                Q.pop();
                for (auto & e : (*G)[u]) {
                    int v = e.first;
                    vector<double> & w = e.second;
                    if (visited.find(v) != visited.end())
                        continue;
                    double x = real_rand(mt_rand);
                    if (x <= w[i]) {
                        visited.insert(v);
                        Q.push(v);
                    }
                }
            }

            S.insert(visited.begin(), visited.end());
        }

        avg += (double) S.size() / (double) N;

    }

    //printf("avg:%f\n", avg);
    parameter->spread = avg;

}

void DTIC_P(int t1, int t2, unordered_set<int> * A, DiGraph * G, int nthreads, int nIters) {

    pthread_t threads[nthreads];
    Parameter parameter[nthreads];

    for (int i = 0; i < nthreads; i++) {
        parameter[i].t1 = t1;
        parameter[i].t2 = t2;
        parameter[i].spread = 0;
        parameter[i].nIters = nIters / nthreads;
        parameter[i].A = A;
        parameter[i].G = G;
    }

    for (int i = 0; i < nthreads; i++) {
        int iret = pthread_create(&threads[i], NULL, simulationThread, (void*) &parameter[i]);
        if (iret) {
            fprintf(stderr, "Error - pthread_create() return code: %d\n", iret);
            exit(EXIT_FAILURE);
        }
    }

    double avgSpread = 0;

    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
        avgSpread += parameter[i].spread;
    }

    avgSpread /= (double) nthreads;

    printf("Spread: %f\n", avgSpread);

}

vector<int> DTIC3(int t1, int t2, unordered_set<int> & SI, unordered_set<int> & TI1, unordered_set<int> & TI2, DiGraph & G) {

    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 mt_rand(seed);
    uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

    unordered_set<int> visited, S;
    queue<int> Q;

    S.insert(SI.begin(), SI.end());
    int nintersect1 = 0;
    int nintersect2 = 0;

    for (int i = t1; i <= t2; i++) {

        visited.clear();

        for (int u : S) {
            visited.insert(u);
            Q.push(u);
        }

        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();
            for (auto e : G[u]) {
                int v = e.first;
                vector<double> & w = e.second;
                if (visited.find(v) != visited.end())
                    continue;
                double x = real_rand(mt_rand);
                if (x <= w[i]) {
                    visited.insert(v);
                    Q.push(v);
                }
            }
        }
        S.insert(visited.begin(), visited.end());
    }

    for (auto e : S) {
        if (TI1.find(e) != TI1.end()) {
            nintersect1++;
        }

        if (TI2.find(e) != TI2.end()) {
            nintersect2++;
        }
    }

    vector<int> v;
    v.push_back(nintersect1);
    v.push_back(nintersect2);

    return v;
}


double greedIC(unordered_set<int> selectSeed, int candidateNode, int iterNum, int t1, int t2, DiGraph & G){

	double spread = 0.0;
	long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	mt19937 mt_rand(seed);
	uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

	for(int i = 0; i < iterNum; i++)
	{
		unordered_set<int> visited, S;
		queue<int> Q;
		visited.clear();
		S.clear();

		S.insert(selectSeed.begin(), selectSeed.end());
		S.insert(candidateNode);

		for(int j = t1; j<=t2; j++)
		{
			visited.clear();
			for (int u : S) {
				visited.insert(u);
			    Q.push(u);
			}

			 unordered_set<int>::iterator it;
//			 printf("greedy weight:");
			 while (!Q.empty()) {
				 int u = Q.front();
			     Q.pop();
			     for (auto e : G[u]) {
			    	 int v = e.first;
			         vector<double> & w = e.second;
//			         printf(" w[i] = %f, ", w[i]);
			         if (visited.find(v) != visited.end())
			         continue;
			         double x = real_rand(mt_rand);
			         if (x <= w[i]) {
			        	 visited.insert(v);
			             Q.push(v);
			         }
			     }
			 }
			 S.insert(visited.begin(), visited.end());
//			 printf("\n");
		}
		spread = spread + S.size();
	}

	spread = spread/iterNum;

//	printf("greedy average spread:%f\n", spread);
	return spread;
}

map<int, map<int,unordered_set<int>>> longgreedy_solution_set(int nodeNum, int selectK, int batchNum, int sampleNum, int t1, int t2, DiGraph & G){
	unordered_set<int> S_set;//all seed
	map<int, unordered_set<int>> K_S_set;// 10-batch seed
	map<int, map<int,unordered_set<int>>> greedy_batchSet;
	unordered_set<int> temp_S_set;
	int batchKNum = selectK/batchNum;


	for(int t= 0; t < t2; t++)
	{
		printf("greed t = %d\n", t);
		map<int,unordered_set<int>> largestK_set = greedy_solution_set(nodeNum, selectK, batchNum, sampleNum, t, t, G);
		printf("greedy largestK_set.keys size = %d", largestK_set.size());
		greedy_batchSet[t] =  largestK_set;
	}
//	printf("greedy greedy_batchSet.keys size = %d", greedy_batchSet.size());
	return greedy_batchSet;
}


map<int,unordered_set<int>> greedy_solution_set(int nodeNum, int selectK, int batchNum, int sampleNum, int t1, int t2, DiGraph & G){


	unordered_set<int> S_set;//all seed
	map<int, unordered_set<int>> K_S_set;// 10-batch seed
	unordered_set<int> temp_S_set;
	for(int i = 0; i<selectK; i++)
	{
		printf("selectK = %d, i = %d, t1 = %d, t2 = %d \n", selectK, i, t1, t2);
		//select top-k nodes greedy
		double best_spread = 0.0;
//		printf("to select the %d-th top node. \n", i);
		int selectNode=0;
		for(int j = 0; j < nodeNum;j++)
		{
//			printf("greedy to check the %d-th node, t1= %d, t2=%d \n", j,t1,t2);
			if(S_set.find(j)!=S_set.end())
			{
				continue;
			}

			double temp_spread = greedIC(S_set, j, sampleNum, t1, t2, G);
			if (temp_spread > best_spread){
				best_spread = temp_spread;
				selectNode = j;
			}

		}
		S_set.insert(selectNode);
//		printf("best_spread:%f, S_set.size() = %d\n", best_spread,S_set.size());
//		printf("selectNode = %d\n", selectNode);
//		printf("i = %d, batchNum = %d\n", i, batchNum);
		if ((i+1)%batchNum == 0){
			temp_S_set.insert(selectNode);
			int mapNum = i+1;
			K_S_set[mapNum] = temp_S_set;
//			printf("K_S_set, mapNum=%d, temp_S_set.size() = %d\n", mapNum, temp_S_set.size());
			temp_S_set.clear();

		}else{
			temp_S_set.insert(selectNode);
		}
//		printf("temp_S_set.size() = %d\n", temp_S_set.size());
	}

	return K_S_set;
}


//vector<int> onlydynamic_DTIC(int t1, int t2,unordered_set<int> & seed_S, map<int, unordered_set<int>>& DYNAMIC_set,DiGraph & G){
//
//}
vector<int> allonlydynamic_DTIC(int t1, int t2,unordered_set<int> & seed_S, unordered_set<int> & TI1, unordered_set<int> & TI2, unordered_set<int> & TI3, map<int, unordered_set<int>> &IterrandomMethod, map<int, unordered_set<int>>& DYNAMIC_set,DiGraph & G)
{
	long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
		mt19937 mt_rand(seed);
		uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

		unordered_set<int> visited, S;
		queue<int> Q;
		S.insert(seed_S.begin(), seed_S.end());

		vector<int> countList;
		countList.push_back(0);
		countList.push_back(0);
		countList.push_back(0);
		countList.push_back(0);

		for(int i=0; i <t2; i++)
		{
			countList.push_back(0);
		}

		for (int i = t1; i <= t2; i++) {

			visited.clear();

			for (int u : S) {
				visited.insert(u);
				Q.push(u);
			}

			while (!Q.empty()) {
				int u = Q.front();
				Q.pop();
				for (auto e : G[u]) {
					int v = e.first;
					vector<double> & w = e.second;
					if (visited.find(v) != visited.end())
						continue;
					double x = real_rand(mt_rand);
					if (x <= w[i]) {
						visited.insert(v);
						Q.push(v);
					}
				}
			}
			S.insert(visited.begin(), visited.end());
		}

		for (auto e : S) {
			if (TI1.find(e) != TI1.end()) {
				countList[0]=countList[0]+1;
			}

			if (TI2.find(e) != TI2.end()) {
				countList[1]=countList[1]+1;
			}

			if (TI3.find(e) != TI3.end()) {
				countList[2]=countList[2]+1;
			}


			map<int, unordered_set<int>>::iterator iter;
			iter = IterrandomMethod.begin();
		   while(iter != IterrandomMethod.end()) {
				//   cout << iter->first << " : " << iter->second << endl;
				   unordered_set<int> readlist = iter->second;
				   if (readlist.find(e) != readlist.end()) {
					   countList[3]=countList[3]+1;
				   }
				   iter++;
		   }



			for(int i = 4; i < 4+t2; i++)
			{
				unordered_set<int> temp = DYNAMIC_set[i-4];
				if (temp.find(e) != temp.end()) {
					countList[i] = countList[i]+1;
				}
			}

		}
		    int numerKey = IterrandomMethod.size();
		    vector<int> v;
		    countList[3] = int(ceil(countList[3]/numerKey));

		return countList;



}

vector<int> alldynamic_TIMMU_DTIC(int t1, int t2,unordered_set<int> & seed_S, unordered_set<int> & TI1, unordered_set<int> & TI2, unordered_set<int> & TI3, map<int, unordered_set<int>> &IterrandomMethod,unordered_set<int> & TIMMU_set, map<int, unordered_set<int>>& DYNAMIC_set,DiGraph & G)
{
	long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
		mt19937 mt_rand(seed);
		uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

		unordered_set<int> visited, S;
		queue<int> Q;
		S.insert(seed_S.begin(), seed_S.end());

		vector<int> countList;
		countList.push_back(0);
		countList.push_back(0);
		countList.push_back(0);
		countList.push_back(0);
		countList.push_back(0);

		for(int i=0; i <t2; i++)
		{
			countList.push_back(0);
		}

		for (int i = t1; i <= t2; i++) {

			visited.clear();

			for (int u : S) {
				visited.insert(u);
				Q.push(u);
			}

			while (!Q.empty()) {
				int u = Q.front();
				Q.pop();
				for (auto e : G[u]) {
					int v = e.first;
					vector<double> & w = e.second;
					if (visited.find(v) != visited.end())
						continue;
					double x = real_rand(mt_rand);
					if (x <= w[i]) {
						visited.insert(v);
						Q.push(v);
					}
				}
			}
			S.insert(visited.begin(), visited.end());
		}

		for (auto e : S) {
			if (TI1.find(e) != TI1.end()) {
				countList[0]=countList[0]+1;
			}

			if (TI2.find(e) != TI2.end()) {
				countList[1]=countList[1]+1;
			}

			if (TI3.find(e) != TI3.end()) {
				countList[2]=countList[2]+1;
			}


			map<int, unordered_set<int>>::iterator iter;
			iter = IterrandomMethod.begin();
		   while(iter != IterrandomMethod.end()) {
				//   cout << iter->first << " : " << iter->second << endl;
				   unordered_set<int> readlist = iter->second;
				   if (readlist.find(e) != readlist.end()) {
					   countList[3]=countList[3]+1;
				   }
				   iter++;
		   }

			if (TIMMU_set.find(e) != TIMMU_set.end()) {
				countList[4] = countList[4]+1;
			}
//			cout<<"before DTIC = t2"<<t2<<endl;
//			int coutIndex = 0;
			for(int i = 5; i < 5+t2; i++)
			{
//				cout<<"dynamic set t2 = "<<t2<<", i="<<i<<endl;
				unordered_set<int> temp = DYNAMIC_set[i-5];
//				unordered_set<int>::iterator its;
//				cout<<"value = ";
//				for(its = temp.begin(); its!=temp.end(); its++)
//				{
//					cout<<*its<<",";
//				}
//				cout<<endl;

//				cout<<"DYNAMIC_set[i].size()="<<DYNAMIC_set[i].size()<<", countList[i] =" <<countList[i] <<endl;
				if (temp.find(e) != temp.end()) {
					countList[i] = countList[i]+1;
				}
			}

		}
		    int numerKey = IterrandomMethod.size();
		    vector<int> v;
		    countList[3] = int(ceil(countList[3]/numerKey));

		return countList;

}


vector<int> dynamic_TIMMU_DTIC(int t1, int t2,unordered_set<int> & seed_S, unordered_set<int> & TIMMU_set, map<int, unordered_set<int>>& DYNAMIC_set,DiGraph & G){
	long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	mt19937 mt_rand(seed);
	uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

	unordered_set<int> visited, S;
	queue<int> Q;
	S.insert(seed_S.begin(), seed_S.end());

	vector<int> countList;
	countList.push_back(0);
	for(int i=0; i <t2; i++)
	{
		countList.push_back(0);
	}

	for (int i = t1; i <= t2; i++) {

		visited.clear();

		for (int u : S) {
			visited.insert(u);
			Q.push(u);
		}

		while (!Q.empty()) {
			int u = Q.front();
			Q.pop();
			for (auto e : G[u]) {
				int v = e.first;
				vector<double> & w = e.second;
				if (visited.find(v) != visited.end())
					continue;
				double x = real_rand(mt_rand);
				if (x <= w[i]) {
					visited.insert(v);
					Q.push(v);
				}
			}
		}
		S.insert(visited.begin(), visited.end());
	}

	for (auto e : S) {
		if (TIMMU_set.find(e) != TIMMU_set.end()) {
			countList[0] = countList[0]+1;
		}

		for(int i = 0; i < t2; i++)
		{
			unordered_set<int> temp = DYNAMIC_set[i];
			if (temp.find(e) != temp.end()) {
				countList[i+1] = countList[i+1]+1;
			}
		}

	}

	return countList;
}

vector<int> greedy_DTIC(int t1, int t2,unordered_set<int> & seed_S, unordered_set<int> & rsm, unordered_set<int> & esm, unordered_set<int> & maxd, map<int, unordered_set<int>> & greedyone, map<int, unordered_set<int>> &IterrandomMethod,DiGraph & G){
	long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	mt19937 mt_rand(seed);
	uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

	unordered_set<int> visited, S;
	queue<int> Q;

	S.insert(seed_S.begin(), seed_S.end());
	vector<int> countList;
	countList.push_back(0);
	countList.push_back(0);
	countList.push_back(0);
	for(int i=0; i <t2; i++)
	{
		countList.push_back(0);
	}
	countList.push_back(0);

	for (int i = t1; i <= t2; i++) {
		visited.clear();
		for (int u : S) {
			visited.insert(u);
			Q.push(u);
		}
		while (!Q.empty()) {
			int u = Q.front();
			Q.pop();
			for (auto e : G[u]) {
				int v = e.first;
				vector<double> & w = e.second;
				if (visited.find(v) != visited.end())
					continue;
				double x = real_rand(mt_rand);
				if (x <= w[i]) {
					visited.insert(v);
					Q.push(v);
				}
			}
		}
		S.insert(visited.begin(), visited.end());
	}

	for (auto e : S) {

		if (rsm.find(e) != rsm.end()) {
			countList[0] = countList[0] + 1;
		}

		if (esm.find(e) != esm.end()) {
			countList[1] = countList[1] + 1;
		}

		if (maxd.find(e) != maxd.end()) {
			countList[2] = countList[2] + 1;
		}


		for(int i = 3; i < (3+t2); i++)
		{
			unordered_set<int> temp = greedyone[i-3];
			if (temp.find(e) != temp.end()) {
				countList[i] = countList[i]+1;
			}
		}


		map<int, unordered_set<int>>::iterator iter;
		iter = IterrandomMethod.begin();
	   while(iter != IterrandomMethod.end()) {
		   unordered_set<int> readlist = iter->second;
		   if (readlist.find(e) != readlist.end()) {
			   countList[countList.size()-1]=countList[countList.size()-1]+1 ;
		   }
		   iter++;
	   }
	}
	int numerKey = IterrandomMethod.size();
	countList[countList.size()-1] = (ceil( countList[countList.size()-1]/numerKey));

	return countList;

}


vector<int> allbaseline_DTIC3(int t1, int t2,unordered_set<int> & seed_S, unordered_set<int> & rsm, unordered_set<int> & esm, unordered_set<int> & maxd, unordered_set<int> & greedyone, map<int, unordered_set<int>> &IterrandomMethod,DiGraph & G){

	long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
	    mt19937 mt_rand(seed);
	    uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

	    unordered_set<int> visited, S;
	    queue<int> Q;

	    S.insert(seed_S.begin(), seed_S.end());
	    int nintersect1 = 0;
	    int nintersect2 = 0;
	    int nintersect3 = 0;
	    int nintersect4 = 0;
	    int nintersect5 = 0;

	    for (int i = t1; i <= t2; i++) {

	        visited.clear();

	        for (int u : S) {
	            visited.insert(u);
	            Q.push(u);
	        }

	        while (!Q.empty()) {
	            int u = Q.front();
	            Q.pop();
	            for (auto e : G[u]) {
	                int v = e.first;
	                vector<double> & w = e.second;
	                if (visited.find(v) != visited.end())
	                    continue;
	                double x = real_rand(mt_rand);
	                if (x <= w[i]) {
	                    visited.insert(v);
	                    Q.push(v);
	                }
	            }
	        }
	        S.insert(visited.begin(), visited.end());
	    }

	    for (auto e : S) {
	        if (rsm.find(e) != rsm.end()) {
	            nintersect1++;
	        }

	        if (esm.find(e) != esm.end()) {
	            nintersect2++;
	        }

	        if (maxd.find(e) != maxd.end()) {
	            nintersect3++;
	        }

	        if (greedyone.find(e) != greedyone.end()) {
	        	nintersect4++;
	        }
	        map<int, unordered_set<int>>::iterator iter;
	        iter = IterrandomMethod.begin();
	       while(iter != IterrandomMethod.end()) {
	            //   cout << iter->first << " : " << iter->second << endl;
	               unordered_set<int> readlist = iter->second;
	               if (readlist.find(e) != readlist.end()) {
	                   nintersect5++;
	               }
	               iter++;
	       }
	    }

	    int numerKey = IterrandomMethod.size();
	    vector<int> v;
	    v.push_back(nintersect1);
	    v.push_back(nintersect2);
	    v.push_back(nintersect3);
	    v.push_back(nintersect4);
	    v.push_back(ceil(nintersect5/numerKey));

	    return v;
}

vector<int> new_DTIC3(int t1, int t2, unordered_set<int> & SI, unordered_set<int> & TI1, unordered_set<int> & TI2, unordered_set<int> & TI3, map<int, unordered_set<int>> &IterrandomMethod, DiGraph & G) {

    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 mt_rand(seed);
    uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

    unordered_set<int> visited, S;
    queue<int> Q;

    S.insert(SI.begin(), SI.end());
    int nintersect1 = 0;
    int nintersect2 = 0;
    int nintersect3 = 0;
    int nintersect4 = 0;


    for (int i = t1; i <= t2; i++) {

        visited.clear();

        for (int u : S) {
            visited.insert(u);
            Q.push(u);
        }

        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();
            for (auto e : G[u]) {
                int v = e.first;
                vector<double> & w = e.second;
                if (visited.find(v) != visited.end())
                    continue;
                double x = real_rand(mt_rand);
                if (x <= w[i]) {
                    visited.insert(v);
                    Q.push(v);
                }
            }
        }
        S.insert(visited.begin(), visited.end());
    }

    for (auto e : S) {
        if (TI1.find(e) != TI1.end()) {
            nintersect1++;
        }

        if (TI2.find(e) != TI2.end()) {
            nintersect2++;
        }

        if (TI3.find(e) != TI3.end()) {
            nintersect3++;
        }

//        if (TI4.find(e) != TI4.end()) {
//            nintersect4++;
//        }

        map<int, unordered_set<int>>::iterator iter;
        iter = IterrandomMethod.begin();
       while(iter != IterrandomMethod.end()) {
            //   cout << iter->first << " : " << iter->second << endl;
               unordered_set<int> readlist = iter->second;
               if (readlist.find(e) != readlist.end()) {
                   nintersect4++;
               }
               iter++;
       }
    }

    int numerKey = IterrandomMethod.size();
    vector<int> v;
    v.push_back(nintersect1);
    v.push_back(nintersect2);
    v.push_back(nintersect3);
    v.push_back(ceil(nintersect4/numerKey));

    return v;
}
vector<int> new_DTIC4(int t1, int t2, unordered_set<int> & SI, unordered_set<int> & TI1, DiGraph & G) {

    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 mt_rand(seed);
    uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

    unordered_set<int> visited, S;
    queue<int> Q;

    S.insert(SI.begin(), SI.end());
    int nintersect1 = 0;
    int nintersect2 = 0;
    int nintersect3 = 0;
    int nintersect4 = 0;


    for (int i = t1; i <= t2; i++) {

        visited.clear();

        for (int u : S) {
            visited.insert(u);
            Q.push(u);
        }

        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();
            for (auto e : G[u]) {
                int v = e.first;
                vector<double> & w = e.second;
                if (visited.find(v) != visited.end())
                    continue;
                double x = real_rand(mt_rand);
                if (x <= w[i]) {
                    visited.insert(v);
                    Q.push(v);
                }
            }
        }
        S.insert(visited.begin(), visited.end());
    }

    for (auto e : S) {
        if (TI1.find(e) != TI1.end()) {
            nintersect1++;
        }

    }

    vector<int> v;
    v.push_back(nintersect1);

    return v;
}

int DTIC2(int t1, int t2, unordered_set<int> & SI, unordered_set<int> & TI, DiGraph & G) {

    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 mt_rand(seed);
    uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);

    unordered_set<int> visited, S;
    queue<int> Q;

    S.insert(SI.begin(), SI.end());
    int nintersect = 0;

    for (int i = t1; i <= t2; i++) {

        visited.clear();

        for (int u : S) {
            visited.insert(u);
            Q.push(u);
        }

        while (!Q.empty()) {
            int u = Q.front();
            Q.pop();
            for (auto e : G[u]) {
                int v = e.first;
                vector<double> & w = e.second;
                if (visited.find(v) != visited.end())
                    continue;
                double x = real_rand(mt_rand);
                if (x <= w[i]) {
                    visited.insert(v);
                    Q.push(v);

                }
            }
        }
        S.insert(visited.begin(), visited.end());
    }

    for (auto e : S) {
        if (TI.find(e) != TI.end()) {
            nintersect++;
        }
    }

    return nintersect;
}

double DTIC(int t1, int t2, unordered_set<int> & A, DiGraph & G, int iterNum) {

    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 mt_rand(seed);
    uniform_real_distribution<double> real_rand = uniform_real_distribution<double>(0, 1.0);
    //    	unordered_set<int> A_val = *it_A;
    //
    //    	unordered_set<int>::iterator it;
    //    	for(it = A_val.begin();it!=A_val.end();it++){
    //    	    printf("elemt = %d, ", *it);
    //    	}
    //    	printf("\n");


    double avg = 0;
    int niter = iterNum;

#pragma omp parallel for reduction(+:avg)
    for (int n = 0; n < niter; n++) {
    	if(n%1000==0)
    	{
    		printf("ValNum = %d\n", n);
    	}

        unordered_set<int> visited, S;
        queue<int> Q;

        visited.clear();
        S.clear();
        
        S.insert(A.begin(), A.end());

        for (int i = t1; i <= t2; i++) {

            visited.clear();

            for (int u : S) {
                visited.insert(u);
                Q.push(u);
            }

            unordered_set<int>::iterator it;
//            for(it = A.begin();it!=A.end();it++){
//            	printf("elemt = %d, ", *it);
//            }
//            printf("\n");

            while (!Q.empty()) {
                int u = Q.front();
                Q.pop();
                for (auto e : G[u]) {
                    int v = e.first;
                    vector<double> & w = e.second;
                    if (visited.find(v) != visited.end())
                        continue;
                    double x = real_rand(mt_rand);

                    if (x <= w[i]) {
//                    	printf("x prob = %f, w[i] prob = %f\n", x, w[i]);
                        visited.insert(v);
                        Q.push(v);
                    }
                }
            }

            S.insert(visited.begin(), visited.end());
        }

        avg += S.size();

    }

    avg /= niter;

    printf("avg:%f\n", avg);
    return avg;
}
