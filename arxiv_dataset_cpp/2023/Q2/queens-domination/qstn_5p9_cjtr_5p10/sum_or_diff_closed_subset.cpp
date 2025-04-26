#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

typedef pair<int,int> pii;
typedef pair<pii,int> piii;

const int max_e=50+5;

void print_usage(char* argv[])
{
	cout<<"Generates all subsets of S of {0,...,e-1} such that for all (x,y) in S\n";
    cout<<"either x+y (mod e) or x-y (mod e) belongs to S.\n\n";
    cout<<"Usage: " <<argv[0]<<" parity e_lb e_ub\n";
    cout<<"\tparity: one of even, odd or all\n";
    cout<<"\te_lb: lower bound on e (integer value >= 1)\n";
    cout<<"\te_ub: upper bound on e (integer value >= e_lb)\n";
}

int check_symmetry(vector<int>& s, int e)
{
	bool ins[max_e+1];
	fill(ins,ins+e,false);
	ins[e]=true;
	for(int i=0; i<((int)s.size()); ++i)
		ins[s[i]]=true;
	int c=0;
	if(not ins[0])
		++c;
	for(int i=0; i<((int)s.size()); ++i)
		if(not ins[e-s[i]])
			++c;
	return c;
}

void print_notables(vector<piii>& notables)
{
	if(notables.empty())
		cout<<"\tNone\n";
	int pe=-1;
	for(int x=0; x<(int)notables.size(); ++x)
	{
		int e=notables[x].first.first;
		int mask=notables[x].first.second;
		int asymmetry_count=notables[x].second;

		if(e!=pe)
			cout<<"\te: "<<e<<"\n";
		cout<<"\t\tS u {e}: ";
		for(int b=0; b<e; ++b)
			if((mask>>b)&1)
				cout<<b<<" ";
		cout<<e<<"\n";
		cout<<"\t\t\tAsymmetry count of S u {e}: "<<asymmetry_count<<"\n";
		pe=e;
	}
}

int main(int argc, char* argv[])
{
	string parity;
	int e_lb,e_ub;
	vector<int> s;
	bool ins[max_e];
	vector<piii> even_notables,odd_notables;

    if(argc!=4)
    {
        print_usage(argv);
        return 1;
    }
    else
    {
    	parity=argv[1];
    	e_lb=atoi(argv[2]);
    	e_ub=atoi(argv[3]);

    	if(parity!="even" and parity!="odd" and parity!="all")
    	{
    		print_usage(argv);
        	return 1;
    	}
    	if(e_lb<1 or e_ub<e_lb)
    	{
    		print_usage(argv);
        	return 1;
    	}
    }


	for(int e=e_lb; e<=e_ub; ++e)
	{
		cout<<"e="<<e<<"\n"<<flush;
		for(int mask=0; mask<(1<<e); ++mask)
		{
			s.clear();
			fill(ins,ins+e,false);
			for(int b=0; b<e; ++b)
				if((mask>>b)&1)
				{
					s.push_back(b);
					ins[b]=true;
				}

			if(parity!="all" and (((((int)s.size())&1)==0) xor parity=="even"))
				continue;

			bool valid=true;
			for(int i=0; i<(int)s.size(); ++i)
			{
				for(int j=0; j<(int)s.size(); ++j)
					if((not ins[(s[i]+s[j])%e]) and (not ins[(s[i]-s[j]+e)%e]))
					{
						valid=false;
						break;
					}
				if(not valid)
					break;
			}

			if(not valid)
				continue;

			cout<<"\tS u {e}: ";
			for(int i=0; i<(int)s.size(); ++i)
				cout<<s[i]<<" ";
			cout<<e<<"\n";

			if(((int)s.size())&1)
			{
				int asymmetry_count=check_symmetry(s,e);
				cout<<"\t\tIs S u {e} symmetric? "<<(asymmetry_count==0 ? "Yes" : "No")<<"\n";
				cout<<"\t\tDoes S contain 0? "<<(s[0]==0 ? "Yes" : "No")<<"\n";
				if(asymmetry_count>0 or s[0]!=0)
					odd_notables.push_back(piii(pii(e,mask),asymmetry_count));
				
			}
			else
			{
				int asymmetry_count=check_symmetry(s,e);
				cout<<"\t\tIs S u {e} almost symmetric? "<<(asymmetry_count<=1 ? "Yes" : "No")<<"\n";
				cout<<"\t\tAsymmetry count of S u {e}: "<<asymmetry_count<<"\n";
				if(asymmetry_count>1)
					even_notables.push_back(piii(pii(e,mask),asymmetry_count));
			}
		}
	}

	cout<<"\n\n";
	if(parity!="odd")
	{
		cout<<"even-sized notables:\n";
		print_notables(even_notables);
	}
	if(parity!="even")
	{
		cout<<"odd-sized notables:\n";
		print_notables(odd_notables);
	}

	return 0;
}