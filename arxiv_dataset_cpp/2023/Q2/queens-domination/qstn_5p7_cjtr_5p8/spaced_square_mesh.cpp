#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

typedef pair<int,int> pii;
typedef pair<vector<int>,vector<int> > pvv;

const int max_p=50+5;
const int max_e=2*max_p;

void print_usage(char* argv[])
{
	cerr<<"Generates all square spaced grids S with equal (p) rows\n";
	cerr<<"and cols of {0,...,e-1} such that for all (x,y) in S\n";
	cerr<<"either x+y (mod e) or x-y (mod e) belongs to S.\n\n";
	cerr<<"Usage: " <<argv[0]<<" parity p_lb p_ub q\n";
	cerr<<"Relevant for (4k+3)*(4k+3) QDP: parity=odd, q=2\n";
	cerr<<"\tparity: one of even, odd or all\n";
	cerr<<"\tp_lb: lower bound on p (integer value >= 1)\n";
	cerr<<"\tp_ub: upper bound on p (integer value >= p_lb)\n";
	cerr<<"\tq: we consider all e such that p<=e<=q*p (integer value >= 2)\n";
}

int p;
char mode;
vector<vector<int> > spacings;
vector<pvv> spacing_pairs;
vector<int> spacing;

void backtrack_spacing(int i, int e, int x)
{
	/*
		i filled of p total
		last (p'th element) is e
		x is the next decision to be made
	*/

	if(x==e)
	{
		assert(i==p);

		spacing.push_back(e);
		spacings.push_back(spacing);
		assert((int)spacing.size()==p+1);
		spacing.pop_back();
	}

	if(i<p)
	{
		spacing.push_back(x);
		backtrack_spacing(i+1,e,x+1);
		spacing.pop_back();
	}

	if(p-i<(e-1)-x+1)
		backtrack_spacing(i,e,x+1);
}

pii intersect_sum_diagonal(int i, int j, int e)
{
	/*
		return (ri,rj)
		ri: row for which sum diagonal at (i,j) intersects j=0 or j=e
			0 if it intersects j=0 at e and j=e at 0
		rj: col for which sum diagonal at (i,j) intersects i=0 or i=e
			0 if it intersects i=0 at e and i=e at 0
	*/

	if(i+j==e)
		return pii(0,0);

	int ri = (i+j<e ? i+j : i+j-e);
	int rj = ri;
	return pii(ri,rj);
}

pii intersect_difference_diagonal(int i, int j, int e)
{
	/*
		return (ri,rj)
		ri: row for which difference diagonal at (i,j) intersects j=0 or j=e
			0 if it intersects j=0 at 0 and j=e at e
		rj: col for which difference diagonal at (i,j) intersects i=0 or i=e
			0 if it intersects i=0 at 0 and i=e at e
	*/

	if(i-j==0)
		return pii(0,0);

	int ri = (i-j>0 ? i-j : i-j+e);
	int rj = e-ri;
	return pii(ri,rj);
}

bool in_hspacing[max_e];
bool in_vspacing[max_e];

int number_of_symmetries(int p, int e, vector<int>& hspacing, vector<int>& vspacing)
{
	bool sym0=true;
	for(int x=0; x<=p; ++x)
		if(hspacing[x]!=vspacing[x])
		{
			sym0=false;
			break;
		}
	bool sym1=true;
	for(int x=0; x<=p; ++x)
		if(hspacing[x]+vspacing[p-x]!=e)
		{
			sym1=false;
			break;
		}
	return ((int)(sym0))+((int)(sym1));
}

int main(int argc, char* argv[])
{
	string parity;
	int p_lb,p_ub;
	int q;

	if(argc!=5)
    {
        print_usage(argv);
        return 1;
    }
    else
    {
    	parity=argv[1];
    	p_lb=atoi(argv[2]);
    	p_ub=atoi(argv[3]);
    	q=atoi(argv[4]);

    	if(parity!="even" and parity!="odd" and parity!="all")
    	{
    		print_usage(argv);
        	return 1;
    	}
    	if(p_lb<1 or p_ub<p_lb)
    	{
    		print_usage(argv);
        	return 1;
    	}
    	if(q<2)
    	{
    		print_usage(argv);
        	return 1;
    	}
    }

    for(p=p_lb; p<=p_ub; ++p)
    {
    	if(parity=="even" and p%2==1)
    		continue;
    	if(parity=="odd" and p%2==0)
    		continue;
		for(int e=p; e<=q*p; ++e)
		{
			spacings.clear();
			spacing.clear();
			spacing.push_back(0);
			backtrack_spacing(1,e,1);

			cout<<"p: "<<p<<"\n";
			cout<<"e: "<<e<<"\n";
			cout<<"# spacings: "<<spacings.size()<<"\n"<<flush;
			/*for(int x=0; x<(int)spacings.size(); ++x)
			{
				cout<<"\t";
				for(int y=0; y<(int)spacings[x].size(); ++y)
					cout<<spacings[x][y]<<" ";
				cout<<"\n";
			}*/

			spacing_pairs.clear();
			for(int hx=0; hx<(int)spacings.size(); ++hx)
				for(int vx=0; vx<(int)spacings.size(); ++vx)
				{
					vector<int> hspacing=spacings[hx];
					vector<int> vspacing=spacings[vx];

					fill(in_hspacing,in_hspacing+e+1,false);
					for(int hy=0; hy<p+1; ++hy)
						in_hspacing[hspacing[hy]]=true;
					fill(in_vspacing,in_vspacing+e+1,false);
					for(int vy=0; vy<p+1; ++vy)
						in_vspacing[vspacing[vy]]=true;

					bool valid=true;
					for(int hy=0; hy<p+1; ++hy)
					{
						for(int vy=0; vy<p+1; ++vy)
						{
							int i=hspacing[hy],j=vspacing[vy];
							pii sij=intersect_sum_diagonal(i,j,e);
							pii dij=intersect_difference_diagonal(i,j,e);

							if((in_hspacing[sij.first] and in_vspacing[sij.second]) or (in_hspacing[dij.first] and in_vspacing[dij.second]))
								continue;
							else
							{
								valid=false;
								break;
							}
						}
						if(not valid)
							break;
					}

					if(valid)
						spacing_pairs.push_back(pvv(hspacing,vspacing));
				}

			cout<<"# spacing pairs: "<<spacing_pairs.size()<<"\n"<<flush;
			for(int x=0; x<(int)spacing_pairs.size(); ++x)
			{
				vector<int> hspacing=spacing_pairs[x].first;
				vector<int> vspacing=spacing_pairs[x].second;

				cout<<"\thspacing: ";
				for(int y=0; y<p+1; ++y)
					cout<<hspacing[y]<<" ";

				cout<<"\tvspacing: ";
				for(int y=0; y<p+1; ++y)
					cout<<vspacing[y]<<" ";
				cout<<"\n";

				cout<<"\t\tnumber of symmetries: "<<number_of_symmetries(p,e,hspacing,vspacing)<<"\n";
			}
		}
	}
}