#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <Eigen/IterativeLinearSolvers>
#include <stdio.h>
#include <emscripten.h>
#include <emscripten/html5.h>
#include <SDL/SDL.h>
#include <queue>

#define PI 3.141592653589793238462

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::VectorXd Vec;
typedef Eigen::Triplet<double> Trip;
typedef Eigen::MatrixXd Mat;

const int DISPPIXELS = 700;


// precomputable quantities
#define POLYMAX 3
double weights[22];
double coords[22];
double normLegendreDerivProducts[POLYMAX+1][POLYMAX+1];
double normLegendreAltProducts[POLYMAX+1][POLYMAX+1];
double normLegendreLeftVals[POLYMAX+1];
double normLegendreRightVals[POLYMAX+1];
double normLegendreDerivLeftVals[POLYMAX+1];
double normLegendreDerivRightVals[POLYMAX+1];
// double normLegendreTripleProducts[POLYMAX+1][POLYMAX+1][POLYMAX+1];
// double normLegendreDerivTripleProducts[POLYMAX+1][POLYMAX+1][POLYMAX+1];

class NSSolve
{
private:
	int N;
	int K;
	int dof;
	double L;
	double epsilon = -1.0;
	double diffconst = 1.0;
	double sigma0;
	double beta0 = 1.0;
	SpMat A;
	SpMat UXP;
	SpMat UXM;
	SpMat UYP;
	SpMat UYM;
	SpMat R;
	SpMat D;
	SpMat C;
	Vec rhs;
	bool steadyState;
	Eigen::BiCGSTAB<SpMat,Eigen::IncompleteLUT<double>> solver;
	void BuildMatA();
	void BuildMatUXP(Vec& ux);
	void BuildMatUXM(Vec& ux);
	void BuildMatUYP(Vec& uy);
	void BuildMatUYM(Vec& uy);
public:
	NSSolve(int N,int K, double L, bool steadyState) : N(N), K(K), L(L), steadyState(steadyState), dof(((N*N*(K+1)*(K+1)))), sigma0((K+1)*(K+2)*4+1)
	{}
	void init()
	{
		A.resize(dof,dof);
		UXP.resize(dof,dof);
		UXM.resize(dof,dof);
		UYP.resize(dof,dof);
		UYM.resize(dof,dof);
		R.resize(dof,dof);
		D.resize(dof,dof);
		C.resize(dof,dof);
		rhs.resize(dof);
		phi.resize(dof);
		for(int i = 0; i < dof; i++) phi(i) = 0;
		BuildMatA();
	}
	Vec phi;
	inline int idx(int ix, int iy, int px, int py) { return (((K+1)*(K+1)*(N*((ix+N)%N)+(iy+N)%N)) + px*(K+1)+py); }
	double Eval(double x, double y);
	void SetRHS(Vec rhs)
	{
		this->rhs = rhs;
		if(steadyState) this->rhs(0)=0.0;
	}
	void SetU(Vec& ux, Vec& uy) 
	{ 
		BuildMatUXP(ux);
		BuildMatUXM(ux);
		BuildMatUYP(uy);
		BuildMatUYM(uy);
	}
	void BuildRHS(double xx = 0.2, double yy = 0.8);
	void Compute(bool useU = false) 
	{
		R=useU?(A+UXP+UXM+UYP+UYM):A;
		R.makeCompressed(); 
		solver.compute(R); 
	}
	double SolResid();
	void Solve()
	{
		phi = solver.solveWithGuess(rhs,phi);
	}
	void PrepEvolve(double dt);
	void Evolve(double dt);
};


void MakeWeights()
{
	weights[0] = 0.1392518728556320;
	coords[0] = -0.0697392733197222;
	weights[1] = 0.1392518728556320;
	coords[1] = 0.0697392733197222;
	weights[2] = 0.1365414983460152;
	coords[2] = -0.2078604266882213;
	weights[3] = 0.1365414983460152;
	coords[3] = 0.2078604266882213;
	weights[4] = 0.1311735047870624;
	coords[4] = -0.3419358208920842;
	weights[5] = 0.1311735047870624;
	coords[5] = 0.3419358208920842;
	weights[6] = 0.1232523768105124;
	coords[6] = -0.4693558379867570;
	weights[7] = 0.1232523768105124;
	coords[7] = 0.4693558379867570;
	weights[8] = 0.1129322960805392;
	coords[8] = -0.5876404035069116;
	weights[9] = 0.1129322960805392;
	coords[9] = 0.5876404035069116;
	weights[10] = 0.1004141444428810;
	coords[10] = -0.6944872631866827;
	weights[11] = 0.1004141444428810;
	coords[11] = 0.6944872631866827;
	weights[12] = 0.0859416062170677;
	coords[12] = -0.7878168059792081;
	weights[13] = 0.0859416062170677;
	coords[13] = 0.7878168059792081;
	weights[14] = 0.0697964684245205;
	coords[14] = -0.8658125777203002;
	weights[15] = 0.0697964684245205;
	coords[15] = 0.8658125777203002;
	weights[16] = 0.0522933351526833;
	coords[16] = -0.9269567721871740;
	weights[17] = 0.0522933351526833;
	coords[17] = 0.9269567721871740;
	weights[18] = 0.0337749015848142;
	coords[18] = -0.9700604978354287;
	weights[19] = 0.0337749015848142;
	coords[19] = 0.9700604978354287;
	weights[20] = 0.0146279952982722;
	coords[20] = -0.9942945854823992;
	weights[21] = 0.0146279952982722;
	coords[21] = 0.9942945854823992;
}

double LegendreEval(int p, double y)
{
	if(p == 0) return 1.0;
	if(p == 1) return y;
	double prev = 1.0;
	double cur = y;
	for(int n = 1; n < p; n++)
	{
		double next = ((2.0*n+1.0)*y*cur - n*prev)/(n+1.0);
		prev = cur;
		cur = next;
	}
	return cur;
}

double LegendreDerivEval(int p, double y)
{
	double val = 0.0;
	for(int n = 0; n < p; n++)
	{
		val = (n+1.0)*LegendreEval(n,y) + y*val;
	}
	return val;
}

double LegendreL2Norm(int p)
{
	return std::pow(2.0/(2.0*p+1.0),0.5);
}



double LegendreEvalNorm(int p, double y)
{
	return LegendreEval(p,y) / LegendreL2Norm(p);
}

double LegendreDerivEvalNorm(int p, double y)
{
	return LegendreDerivEval(p,y) / LegendreL2Norm(p);
}

void MakeLegendreEndpointVals()
{
	for(int p = 0; p < POLYMAX+1; p++)
	{
		normLegendreLeftVals[p] = LegendreEvalNorm(p,-1.0);
		normLegendreRightVals[p] = LegendreEvalNorm(p,1.0);
		normLegendreDerivLeftVals[p] = LegendreDerivEvalNorm(p,-1.0);
		normLegendreDerivRightVals[p] = LegendreDerivEvalNorm(p,1.0);
	}
}

void MakeLegendreDerivProducts()
{
	for(int p = 0; p < POLYMAX+1; p++)
	{
		for(int q = 0; q < POLYMAX+1; q++)
		{
			double res = 0.0;
			for(int k = 0; k < 22; k++)
			{
				res += weights[k] * LegendreDerivEvalNorm(p,coords[k]) * LegendreDerivEvalNorm(q,coords[k]);
			}
			normLegendreDerivProducts[p][q] = res;
		}
	}
}

void MakeLegendreAltProducts()
{
	for(int p = 0; p < POLYMAX+1; p++)
	{
		for(int q = 0; q < POLYMAX+1; q++)
		{
			double res = 0.0;
			for(int k = 0; k < 22; k++)
			{
				res += weights[k] * LegendreEvalNorm(p,coords[k]) * LegendreDerivEvalNorm(q,coords[k]);
			}
			normLegendreAltProducts[p][q] = res;
		}
	}
}

// void MakeLegendreTripleProducts()
// {
// 	for(int p = 0; p < POLYMAX+1; p++)
// 	{
// 		for(int q = 0; q < POLYMAX+1; q++)
// 		{
// 			for(int r = 0; r < POLYMAX+1; r++)
// 			{
// 				double res = 0.0;
// 				for(int k = 0; k < 22; k++)
// 				{
// 					res += weights[k] * LegendreEvalNorm(p,coords[k]) * LegendreEvalNorm(q,coords[k]) * LegendreEvalNorm(r,coords[k]);
// 				}
// 				normLegendreTripleProducts[p][q][r] = res;
// 			}
// 		}
// 	}
// }

// void MakeLegendreDerivTripleProducts()
// {
// 	for(int p = 0; p < POLYMAX+1; p++)
// 	{
// 		for(int q = 0; q < POLYMAX+1; q++)
// 		{
// 			for(int r = 0; r < POLYMAX+1; r++)
// 			{
// 				double res = 0.0;
// 				for(int k = 0; k < 22; k++)
// 				{
// 					res += weights[k] * LegendreEvalNorm(p,coords[k]) * LegendreEvalNorm(q,coords[k]) * LegendreDerivEvalNorm(r,coords[k]);
// 				}
// 				normLegendreDerivTripleProducts[p][q][r] = res;
// 			}
// 		}
// 	}
// }

void NSSolve::BuildMatA()
{
	double h = L/N;
	double hbeta0 = std::pow(h,beta0);
	std::vector<Trip> elems;

	// Diagonal blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val += diffconst * std::pow(2.0/h,2) * normLegendreDerivProducts[px][qx] * (py == qy ? 1.0 : 0.0);
					val += diffconst * std::pow(2.0/h,2) * (px == qx ? 1.0 : 0.0) * normLegendreDerivProducts[py][qy];
					
					// East
					val += diffconst * (2.0/h) * (2.0/h) * (-0.5) * (py == qy ? 1.0 : 0.0)
						* normLegendreRightVals[qx] * normLegendreDerivRightVals[px];
					val -= (2.0/h) * (2.0/h) *(-1.0)* epsilon * 0.5 * (py == qy ? 1.0 : 0.0)
						* normLegendreRightVals[px] * normLegendreDerivRightVals[qx];
					val += (2.0/h) * (sigma0/hbeta0) * ( normLegendreRightVals[px]*normLegendreRightVals[qx] )
						* (py == qy ? 1.0 : 0.0);
					
					// West
					val += diffconst * (2.0/h) *(2.0/h) * (0.5) * (py == qy ? 1.0 : 0.0)
						* normLegendreLeftVals[qx] * normLegendreDerivLeftVals[px];
					val -= (2.0/h) *(2.0/h) * epsilon * 0.5 * (py == qy ? 1.0 : 0.0)
						* normLegendreLeftVals[px] * normLegendreDerivLeftVals[qx];
					val += (2.0/h) * (sigma0/hbeta0) * ( normLegendreLeftVals[qx]*normLegendreLeftVals[px] )
						* (py == qy ? 1.0 : 0.0);
					

					// North
					val += diffconst * (2.0/h) *(2.0/h) * (-0.5) * (px == qx ? 1.0 : 0.0)
						* normLegendreRightVals[qy] * normLegendreDerivRightVals[py];
					val -= (2.0/h) * (-1.0)*(2.0/h) * epsilon * 0.5 * (px == qx ? 1.0 : 0.0)
						* normLegendreRightVals[py] * normLegendreDerivRightVals[qy];
					val += (2.0/h) * (sigma0/hbeta0) * ( normLegendreRightVals[py]*normLegendreRightVals[qy] )
						* (px == qx ? 1.0 : 0.0);
					
					// South
					val += diffconst * (2.0/h) *(2.0/h) * (0.5) * (px == qx ? 1.0 : 0.0)
						* normLegendreLeftVals[qy] * normLegendreDerivLeftVals[py];
					val -= (2.0/h) * epsilon * 0.5 *(2.0/h) * (px == qx ? 1.0 : 0.0)
						* normLegendreLeftVals[py] * normLegendreDerivLeftVals[qy];
					val += (2.0/h) * (sigma0/hbeta0) * ( normLegendreLeftVals[qy]*normLegendreLeftVals[py] )
						* (px == qx ? 1.0 : 0.0);
					
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy,px,py);
							Trip t(idxv,idxphi,val);
							if(!steadyState || idxv != 0) elems.push_back(t);
							if(steadyState && idxv == 0 && px == 0 && qx == 0)
							{
								Trip t1(idxv,idxphi,1.0);
								elems.push_back(t1);
							}
						}
					}
				}
			}
		}
	}

	// East blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val += diffconst * (2.0/h) *(2.0/h) * (-0.5) * (py == qy ? 1.0 : 0.0)
						* normLegendreRightVals[qx] * normLegendreDerivLeftVals[px];
					val -= (2.0/h) * epsilon *(2.0/h) * 0.5 * (py == qy ? 1.0 : 0.0)
						* normLegendreLeftVals[px] * normLegendreDerivRightVals[qx];
					val += (2.0/h) * (-1.0 * sigma0/hbeta0) * ( normLegendreLeftVals[px]*normLegendreRightVals[qx] )
						* (py == qy ? 1.0 : 0.0);
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix+1,iy,px,py);
							Trip t(idxv,idxphi,val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	// West blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val += diffconst * (2.0/h) * (0.5) *(2.0/h) * (py == qy ? 1.0 : 0.0)
						* normLegendreLeftVals[qx] * normLegendreDerivRightVals[px];
					val -= (2.0/h) * (-1.0) * epsilon * 0.5 *(2.0/h) * (py == qy ? 1.0 : 0.0)
						* normLegendreRightVals[px] * normLegendreDerivLeftVals[qx];
					val += (2.0/h) * (-1.0 * sigma0/hbeta0) * ( normLegendreLeftVals[qx]*normLegendreRightVals[px] )
						* (py == qy ? 1.0 : 0.0);
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix-1,iy,px,py);
							Trip t(idxv,idxphi,val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	// North blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val += diffconst * (2.0/h) * (-0.5) *(2.0/h) * (px == qx ? 1.0 : 0.0)
						* normLegendreRightVals[qy] * normLegendreDerivLeftVals[py];
					val -= (2.0/h) * epsilon * 0.5 *(2.0/h) * (px == qx ? 1.0 : 0.0)
						* normLegendreLeftVals[py] * normLegendreDerivRightVals[qy];
					val += (2.0/h) * (-1.0 * sigma0/hbeta0) * ( normLegendreLeftVals[py]*normLegendreRightVals[qy] )
						* (px == qx ? 1.0 : 0.0);
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy+1,px,py);
							Trip t(idxv,idxphi,val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	// South blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val += diffconst * (2.0/h) * (0.5) *(2.0/h) * (px == qx ? 1.0 : 0.0)
						* normLegendreLeftVals[qy] * normLegendreDerivRightVals[py];
					val -= (2.0/h) * (-1.0) * epsilon * 0.5 *(2.0/h) * (px == qx ? 1.0 : 0.0)
						* normLegendreRightVals[py] * normLegendreDerivLeftVals[qy];
					val += (2.0/h) * (-1.0 * sigma0/hbeta0) * ( normLegendreLeftVals[qy]*normLegendreRightVals[py] )
						* (px == qx ? 1.0 : 0.0);
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy-1,px,py);
							Trip t(idxv,idxphi,val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	A.setFromTriplets(elems.begin(),elems.end());
}


void NSSolve::BuildMatUXP(Vec& ux)
{
	double h = L/N;
	std::vector<Trip> elems;

	// Diagonal blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val -=  (2.0/h) *normLegendreAltProducts[px][qx] * (py == qy ? 1.0 : 0.0);
					
					val -= (2.0/h) * (-1.0) * normLegendreRightVals[px]*normLegendreRightVals[qx]
						 * ( py == qy ? 1.0 : 0.0 );

					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy,px,py);
							double uxp = ux(idx(ix,iy,0,0));
							uxp = uxp > 0.0 ? uxp : 0.0;
							Trip t(idxv,idxphi,uxp*val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	// West blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val -= (2.0/h) * normLegendreLeftVals[qx]*normLegendreRightVals[px]
						 * ( py == qy ? 1.0 : 0.0 );
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix-1,iy,px,py);
							double uxp = ux(idx(ix,iy,0,0));
							uxp = uxp > 0.0 ? uxp : 0.0;
							Trip t(idxv,idxphi,uxp*val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	UXP.setFromTriplets(elems.begin(),elems.end());
}


void NSSolve::BuildMatUXM(Vec& ux)
{
	double h = L/N;
	std::vector<Trip> elems;

	// Diagonal blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val -= (2.0/h) *normLegendreAltProducts[px][qx] * (py == qy ? 1.0 : 0.0);
					val -= (2.0/h) * normLegendreLeftVals[qx]*normLegendreLeftVals[px]
						 * ( py == qy ? 1.0 : 0.0 );

					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy,px,py);
							double uxm = ux(idx(ix,iy,0,0));
							uxm = uxm < 0.0 ? uxm : 0.0;
							Trip t(idxv,idxphi,uxm*val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	// East blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val -= (2.0/h) * (-1.0) * normLegendreLeftVals[px]*normLegendreRightVals[qx]
						 * ( py == qy ? 1.0 : 0.0 );
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix+1,iy,px,py);
							double uxm = ux(idx(ix,iy,0,0));
							uxm = uxm < 0.0 ? uxm : 0.0;
							Trip t(idxv,idxphi,uxm*val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	UXM.setFromTriplets(elems.begin(),elems.end());
}



void NSSolve::BuildMatUYP(Vec& uy)
{
	double h = L/N;
	std::vector<Trip> elems;

	// Diagonal blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val -=(2.0/h) * (px == qx ? 1.0 : 0.0) * normLegendreAltProducts[py][qy];
					val -= (2.0/h) * (-1.0) * normLegendreRightVals[py]*normLegendreRightVals[qy]
						 * ( px == qx ? 1.0 : 0.0 );

					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy,px,py);
							double uyp = uy(idx(ix,iy,0,0));
							uyp = uyp > 0.0 ? uyp : 0.0;
							Trip t(idxv,idxphi,uyp*val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	// South blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val -= (2.0/h) * normLegendreLeftVals[qy]*normLegendreRightVals[py]
						 * ( px == qx ? 1.0 : 0.0 );
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy-1,px,py);
							double uyp = uy(idx(ix,iy,0,0));
							uyp = uyp > 0.0 ? uyp : 0.0;
							Trip t(idxv,idxphi,uyp*val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	UYP.setFromTriplets(elems.begin(),elems.end());
}


void NSSolve::BuildMatUYM(Vec& uy)
{
	double h = L/N;
	std::vector<Trip> elems;

	// Diagonal blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val -= (2.0/h) *(px == qx ? 1.0 : 0.0) * normLegendreAltProducts[py][qy];
					val -= (2.0/h) * normLegendreLeftVals[qy]*normLegendreLeftVals[py]
						 * ( px == qx ? 1.0 : 0.0 );

					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy,px,py);
							double uym = uy(idx(ix,iy,0,0));
							uym = uym < 0.0 ? uym : 0.0;
							Trip t(idxv,idxphi,uym*val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	// North blocks
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			for(int qx = 0; qx < K+1; qx++)
			{
				for(int qy = 0; qy < K+1; qy++)
				{
					double val = 0.0;
					val -= (2.0/h) * (-1.0) * normLegendreLeftVals[py]*normLegendreRightVals[qy]
						 * ( px == qx ? 1.0 : 0.0 );
					for(int ix = 0; ix < N; ix++)
					{
						for(int iy = 0; iy < N; iy++)
						{
							int idxv = idx(ix,iy,qx,qy);
							int idxphi = idx(ix,iy+1,px,py);
							double uym = uy(idx(ix,iy,0,0));
							uym = uym < 0.0 ? uym : 0.0;
							Trip t(idxv,idxphi,uym*val);
							if(!steadyState || idxv != 0) elems.push_back(t);
						}
					}
				}
			}
		}
	}

	UYM.setFromTriplets(elems.begin(),elems.end());
}



double PeriodicGaussian(double x, double y, double r)
{
	double val = 0.0;
	for(int i = -2; i <= 2; i++)
	{
		for(int j = -2; j <= 2; j++)
		{
			val += std::exp(-0.5*std::pow((x-1.0*i)/r,2)-0.5*std::pow((y-1.0*j)/r,2));
		}
	}
	return val;
}

double EvalRHS(double x, double y, double xx, double yy)
{ 
	return PeriodicGaussian(x-xx,y-yy,0.15) - PeriodicGaussian(x-(1.0-xx),y-(1.0-yy),0.15);
}

void NSSolve::BuildRHS(double xx, double yy)
{
	double h = L/N;
	for(int ix = 0; ix < N; ix++)
	{
		for(int iy = 0; iy < N; iy++)
		{
			double xc = (ix+0.5)*h;
			double yc = (iy+0.5)*h;
			for(int px = 0; px < K+1; px++)
			{
				for(int py = 0; py < K+1; py++)
				{
					double val = 0.0;
					for(int j = 0; j < 22; j++)
					{
						for(int k = 0; k < 22; k++)
						{
							val += weights[j]*weights[k]
								* LegendreEvalNorm(px,coords[j])
								* LegendreEvalNorm(py,coords[k])
								* EvalRHS((xc+coords[j]*(h/2.0))/L, (yc+coords[k]*(h/2.0))/L, xx, yy);
						}
					}
					rhs(idx(ix,iy,px,py)) = val;
				}
			}
		}
	}
	rhs(0) = 0.0;
}

double NSSolve::Eval(double x, double y)
{
	if(x < 0.0) return Eval(x+L,y);
	if(x>L) return Eval(x-L,y);
	if(y<0.0) return Eval(x,y+L);
	if(y > L) return Eval(x,y-L);
	double h = L/N;
	int ix = x/h;
	int iy = y/h;
	double val = 0.0;
	double xc = (ix+0.5)*h;
	double yc = (iy+0.5)*h;
	for(int px = 0; px < K+1; px++)
	{
		for(int py = 0; py < K+1; py++)
		{
			val += phi(idx(ix,iy,px,py)) * LegendreEvalNorm(px,(x-xc)*(2.0/h)) * LegendreEvalNorm(py,(y-yc)*(2.0/h));
		}
	}
	return val;
}


extern "C" {

Mat disp(DISPPIXELS,DISPPIXELS);
Mat dispTemp(DISPPIXELS,DISPPIXELS);


double len = 10.0;
SDL_Surface *screen;
NSSolve nsSolve(40,1,len,true);
NSSolve nsSolveOmega(40,1,len,false);
std::queue<int> workQueue;
bool nsSolveInited(false);
bool mouseIsDown(false);
bool touchIsStarted(false);

Vec ux(6400); ////
Vec uy(6400); ////

double getVelocityX(long targetX)
{
	return targetX/700.0;
}


double getVelocityY(long targetY)
{
	return targetY/700.0;
}

void mouse_update(const EmscriptenMouseEvent *e)
{
	double ux = getVelocityX(e->targetX);
	double uy = getVelocityY(e->targetY);
	nsSolveOmega.BuildRHS(ux,uy);
}

void touch_update(const EmscriptenTouchEvent *e)
{
	if(e->numTouches > 0)
	{
		long targetX = 0;
		long targetY = 0;
		for(int i = 0; i < e->numTouches; i++)
		{
			targetX += e->touches[i].targetX;
			targetY += e->touches[i].targetY;
		}
		targetX /= e->numTouches;
		targetY /= e->numTouches;
		double ux = getVelocityX(targetX);
		double uy = getVelocityY(targetY);
		nsSolveOmega.BuildRHS(ux,uy);
	}
}

#define NUMLEVELS 13

int getColorIndex(double val)
{
	return std::floor( (NUMLEVELS-1) * ((val+0.0001)/1.0002) );
}

double getLambda(double val, int colorIndex)
{
	return (val-colorIndex/(1.0*NUMLEVELS-1.0))*(1.0*NUMLEVELS-1.0);
}

double red(double lambda, int colorIndex)
{
	return lambda>0.5 ? 1.0 : 254.0;
}

double green(double lambda, int colorIndex)
{
	return lambda>0.5 ? 1.0 : 254.0;
}

double blue(double lambda, int colorIndex)
{
	return lambda>0.5 ? 1.0 : 254.0;
}

void repaint(NSSolve& cd)
{
	double maxphi = cd.Eval(0.0, 0.0);
	double minphi = cd.Eval(0.0,0.0);
	for(int i = 0; i < 100; i++)
	{
		for(int j = 0; j < 100; j++)
		{
			double val = cd.Eval(len*(1.0*j)/100,len*(1.0*i)/100);
			if(val > maxphi) maxphi = val;
			if(val < minphi) minphi = val;
		}
	}

	for(int i = 0; i < DISPPIXELS; i++)
	{
		for(int j = 0; j < DISPPIXELS; j++)
		{
			dispTemp(i,j) = (cd.Eval(len*(1.0*j)/DISPPIXELS,len*(1.0*i)/DISPPIXELS)-minphi)/(maxphi-minphi);
		}
	}

	int shift = 25;

	for(int i = 0; i < DISPPIXELS; i++)
	{
		for(int j = 0; j < DISPPIXELS; j++)
		{
			disp(i,j) = dispTemp((i+shift)%DISPPIXELS,(j+shift)%DISPPIXELS)
				+ dispTemp((i+shift)%DISPPIXELS,j)
				+ dispTemp((i+shift)%DISPPIXELS,(j-shift+DISPPIXELS)%DISPPIXELS)
				+ dispTemp(i,(j-shift+DISPPIXELS)%DISPPIXELS)
				+ dispTemp((i-shift+DISPPIXELS)%DISPPIXELS,(j-shift+DISPPIXELS)%DISPPIXELS)
				+ dispTemp((i-shift+DISPPIXELS)%DISPPIXELS,j)
				+ dispTemp((i-shift+DISPPIXELS)%DISPPIXELS,(j+shift)%DISPPIXELS)
				+ dispTemp(i,(j+shift)%DISPPIXELS)
				+ dispTemp(i,j);
			disp(i,j) /= 9.0;
		}
	}

	if (SDL_MUSTLOCK(screen)) SDL_LockSurface(screen);
	for (int i = 0; i < DISPPIXELS; i++) {
		for (int j = 0; j < DISPPIXELS; j++) {
			double val = disp(i,j);
			int colorIndex = getColorIndex(val);
			double lambda = getLambda(val,colorIndex);
			double valr = red(lambda,colorIndex)*255.0;
			double valg = green(lambda,colorIndex)*255.0;
			double valb = blue(lambda,colorIndex)*255.0;
			*((Uint32*)screen->pixels + i * DISPPIXELS + j) = SDL_MapRGBA(screen->format, (int)valr, (int)valg, (int)valb, 255);
		}
	}
	if (SDL_MUSTLOCK(screen)) SDL_UnlockSurface(screen);
	SDL_Flip(screen); 
}


EM_BOOL mouseclick_callback(int eventType, const EmscriptenMouseEvent *e, void *userData)
{
	mouse_update(e);
	workQueue.push(0);
	workQueue.push(1);
	mouseIsDown = false;
	return 1;
}

EM_BOOL mouseleave_callback(int eventType, const EmscriptenMouseEvent *e, void *userData)
{
	mouseIsDown = false;
	return 1;
}

EM_BOOL mouseup_callback(int eventType, const EmscriptenMouseEvent *e, void *userData)
{
	mouseIsDown = false;
	return 1;
}

EM_BOOL mousedown_callback(int eventType, const EmscriptenMouseEvent *e, void *userData)
{
	mouseIsDown = true;
	return 1;
}

EM_BOOL mousemove_callback(int eventType, const EmscriptenMouseEvent *e, void *userData)
{
	if(mouseIsDown)
	{
		mouse_update(e);
		workQueue.push(0);
	}
	return 1;
}

EM_BOOL touchmove_callback(int eventType, const EmscriptenTouchEvent *e, void *userData)
{
	if(touchIsStarted)
	{
		touch_update(e);
		workQueue.push(0);
	}
	return 1;
}

EM_BOOL touchstart_callback(int eventType, const EmscriptenTouchEvent *e, void *userData)
{
	touchIsStarted = true;
	return 1;
}

EM_BOOL touchend_callback(int eventType, const EmscriptenTouchEvent *e, void *userData)
{
	touchIsStarted = false;
	touch_update(e);
	workQueue.push(0);
	workQueue.push(1);
	return 1;
}

EM_BOOL touchcancel_callback(int eventType, const EmscriptenTouchEvent *e, void *userData)
{
	touchIsStarted = false;
	return 1;
}

int n = 0;
void init()
{
	double dt = 0.001;
	if(n < 5)
	{
		n++;
		return;
	}

	if(!nsSolveInited)
	{
		nsSolve.init();
		nsSolveOmega.init();
		nsSolveOmega.BuildRHS();
		nsSolve.Compute();
		// for(int ix = 0; ix < 40; ix++)
		// {
		// 	for(int iy = 0; iy < 40; iy++)
		// 	{
		// 		nsSolveOmega.phi(nsSolve.idx(ix,iy,0,0)) =
		// 			 (25*std::cos(2.0*PI*ix/40.0)-25*std::cos(2.0*PI*iy/40.0))*2.0*PI/len;
		// 	}
		// }
		nsSolveInited = true;
	}
	nsSolve.SetRHS(nsSolveOmega.phi);
	nsSolve.Solve();
	nsSolveOmega.SetU(ux,uy);
	nsSolveOmega.PrepEvolve(dt);
	nsSolveOmega.Evolve(dt);
	repaint(nsSolve);
	for(int ix = 0; ix < 40; ix++)
	{
		for(int iy = 0; iy < 40; iy++)
		{
			ux(nsSolve.idx(ix,iy,0,0)) = -1.0*nsSolve.phi(nsSolve.idx(ix,iy,0,1));
			uy(nsSolve.idx(ix,iy,0,0)) = nsSolve.phi(nsSolve.idx(ix,iy,1,0));
		}
	}
	return;

}

void NSSolve::PrepEvolve(double dt)
{
	R=A+UXP+UXM+UYP+UYM;
	C.setIdentity();
	C += R * dt/2.0;
	D.setIdentity();
	D -= R * dt/2.0;
	C.makeCompressed();
	solver.compute(C);
}

void NSSolve::Evolve(double dt)
{
	Vec b = rhs*dt+D*phi;
	Vec guess = phi;
	phi = solver.solveWithGuess(b,guess);	
}

int main(int argc, char ** argv)
{
	MakeWeights();
	MakeLegendreDerivProducts();
	MakeLegendreAltProducts();
	MakeLegendreEndpointVals();
	// MakeLegendreTripleProducts();
	// MakeLegendreDerivTripleProducts();

	ux.setZero();
	uy.setZero();
	for(int ix = 0; ix < 40; ix++)
	{
		for(int iy = 0; iy < 40; iy++)
		{
			ux(nsSolve.idx(ix,iy,0,0)) = 25*std::sin(2.0*PI*iy/40.0);
			uy(nsSolve.idx(ix,iy,0,0)) = 25*std::sin(2.0*PI*ix/40.0);
		}
	}
	
	

	workQueue.push(0);
	workQueue.push(1);
	
	SDL_Init(SDL_INIT_VIDEO);
	screen = SDL_SetVideoMode(DISPPIXELS, DISPPIXELS, 32, SDL_SWSURFACE);

	emscripten_set_click_callback("canvas", 0, 1, mouseclick_callback);
	emscripten_set_mousedown_callback("canvas", 0, 1, mousedown_callback);
	emscripten_set_mouseup_callback("canvas", 0, 1, mouseup_callback);
	emscripten_set_mousemove_callback("canvas", 0, 1, mousemove_callback);
	emscripten_set_mouseleave_callback("canvas", 0, 1, mouseleave_callback);
	emscripten_set_touchstart_callback("canvas", 0, 1, touchstart_callback);
	emscripten_set_touchend_callback("canvas", 0, 1, touchend_callback);
	emscripten_set_touchcancel_callback("canvas", 0, 1, touchcancel_callback);
	emscripten_set_touchmove_callback("canvas", 0, 1, touchmove_callback);


	emscripten_set_main_loop(init,0,0);
}

}