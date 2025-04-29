#include "build_hamiltonian.h"

Matrix RotationAndTrap(vector<int>& channels, unsigned int QN)
{
	int n, j1, j2;
	unsigned int basis_size = basisSize(channels, QN);

	Matrix RotationAndTrap(basis_size, basis_size, 0);

	for (unsigned int i = 0; i < basis_size; i++)
	{
		n = channels[(int64_t)QN * (int64_t)i];
		j1 = channels[(int64_t)QN * (int64_t)i + 3];
		j2 = channels[(int64_t)QN * (int64_t)i + 4];
		RotationAndTrap(i) = diagonalFunction(n, j1, j2);
	}

	return RotationAndTrap;
}

Matrix isoInteraction(vector<int>& channels, unsigned int QN, double iso_strength)
{
	int n, nprim;
	unsigned int basis_size = basisSize(channels, QN);

	Matrix isoInteraction(basis_size, basis_size, 0);

	for (unsigned int i = 0; i < basis_size; i++)
	{
		for (unsigned int j = 0; j < basis_size; j++)
		{
			if (differByNOnly(channels, QN, i, j))
			{
				n = channels[(int64_t)QN * (int64_t)i];
				nprim = channels[(int64_t)QN * (int64_t)j];
				isoInteraction(i, j) = deltaFunction(iso_strength, n, nprim);
			}
			else
			{
				continue;
			}
		}
	}

	return isoInteraction;
}

Matrix anisoInteraction(vector<int>& channels, unsigned int QN, double aniso_strength)
{
	int n, nprim;
	unsigned int basis_size = basisSize(channels, QN);

	Matrix anisoInteraction(basis_size, basis_size, 0);

	for (unsigned int i = 0; i < basis_size; i++)
	{
		for (unsigned int j = 0; j < basis_size; j++)
		{
			if (differByNAndjsOnly(channels, QN, i, j) && jsExchangeByOne(channels, QN, i, j) && momentumConserved(channels, QN, i, j))
			{
				n = channels[(int64_t)QN * (int64_t)i];
				nprim = channels[(int64_t)QN * (int64_t)j];
				anisoInteraction(i, j) = deltaFunction(aniso_strength, n, nprim);
			}
			else
			{
				continue;
			}
		}
	}

	return anisoInteraction;
}

Matrix electricField(vector<int>& channels, unsigned int QN, double el_field)
{
	int J, Jprim, M, Mprim, j1, j2, j1prim, j2prim;
	double CG, CGprim;
	unsigned int basis_size = basisSize(channels, QN);

	Matrix electricField(basis_size, basis_size, 0);

	for (unsigned int i = 0; i < basis_size; i++)
	{
		for (unsigned int j = 0; j < basis_size; j++)
		{
			if (sameNAndSpin(channels, QN, i, j) && onejDifferByOne(channels, QN, i, j))
			{
				J = channels[(int64_t)QN * (int64_t)i + 1];
				Jprim = channels[(int64_t)QN * (int64_t)j + 1];
				M = channels[(int64_t)QN * (int64_t)i + 2];
				Mprim = channels[(int64_t)QN * (int64_t)j + 2];
				j1 = channels[(int64_t)QN * (int64_t)i + 3];
				j2 = channels[(int64_t)QN * (int64_t)i + 4];
				j1prim = channels[(int64_t)QN * (int64_t)j + 3];
				j2prim = channels[(int64_t)QN * (int64_t)j + 4];
				
				for (int m1 = -j1; m1 <= j1; m1++)
				{
					for (int m2 = -j2; m2 <= j2; m2++)
					{
						for (int m1prim = -j1prim; m1prim <= j1prim; m1prim++)
						{
							for (int m2prim = -j2prim; m2prim <= j2prim; m2prim++)
							{
								if (nonZeroCG(m1, m2, M, m1prim, m2prim, Mprim) && m1 == m1prim && m2 == m2prim)
								{
									CG = ClebschGordanCoefficient(j1, m1, j2, m2, J, M);
									CGprim = ClebschGordanCoefficient(j1prim, m1prim, j2prim, m2prim, Jprim, Mprim);

									// molecules may be different
									if (j2 == j2prim)
									{
										electricField(i, j) += CG * CGprim * electricFunction(mu1, el_field, j1, j1prim, m1);
									}

									if (j1 == j1prim)
									{
										electricField(i, j) += CG * CGprim * electricFunction(mu2, el_field, j2, j2prim, m2);
									}

								}
								else
								{
									continue;
								}
							}
						}
					}
				}
			}
			else
			{
				continue;
			}
		}
	}

	return electricField; 
}

Matrix magneticField(vector<int>& channels, double mg_field)
{
	double sum;
	unsigned int basis_size = basisSize(channels, QN_withspin);

	Matrix magneticField(basis_size, basis_size, 0);

	for (unsigned int i = 0; i < basis_size; i++)
	{
		sum = (double)channels[(int64_t)QN_withspin * (int64_t)i + 6];

		if (sum != 0)
		{
			magneticField(i) = magneticFunction(mg_field, sum);
		}
		
	}

	return magneticField;
}

Matrix spinRotation(vector<int>& channels, double spinrot_strength)
{
	int J, Jprim, M, Mprim, j1, j2, j1prim, j2prim, Stot, Stotprim, MS, MSprim;
	double ms1, ms1prim, ms2, ms2prim;
	double CG, CGprim, spinCG, spinCGprim;
	unsigned int basis_size = basisSize(channels, QN_withspin);
	
	Matrix spinRotation(basis_size, basis_size, 0);

	for (unsigned int i = 0; i < basis_size; i++)
	{
		for (unsigned int j = 0; j < basis_size; j++)
		{
			if (differByJMjSAndMsOnly(channels, i, j))
			{
				J = channels[(int64_t)QN_withspin * (int64_t)i + 1];
				Jprim = channels[(int64_t)QN_withspin * (int64_t)j + 1];
				M = channels[(int64_t)QN_withspin * (int64_t)i + 2];
				Mprim = channels[(int64_t)QN_withspin * (int64_t)j + 2];
				j1 = channels[(int64_t)QN_withspin * (int64_t)i + 3];
				j2 = channels[(int64_t)QN_withspin * (int64_t)i + 4];
				j1prim = channels[(int64_t)QN_withspin * (int64_t)j + 3];
				j2prim = channels[(int64_t)QN_withspin * (int64_t)j + 4];
				Stot = channels[(int64_t)QN_withspin * (int64_t)i + 5];
				Stotprim = channels[(int64_t)QN_withspin * (int64_t)j + 5];
				MS = channels[(int64_t)QN_withspin * (int64_t)i + 6];
				MSprim = channels[(int64_t)QN_withspin * (int64_t)j + 6];

				for (int m1 = -j1; m1 <= j1; m1++)
				{
					for (int m2 = -j2; m2 <= j2; m2++)
					{
						for (int m1prim = -j1prim; m1prim <= j1prim; m1prim++)
						{
							for (int m2prim = -j2prim; m2prim <= j2prim; m2prim++)
							{
								if (nonZeroCG(m1, m2, M, m1prim, m2prim, Mprim))
								{
									CG = ClebschGordanCoefficient(j1, m1, j2, m2, J, M);
									CGprim = ClebschGordanCoefficient(j1prim, m1prim, j2prim, m2prim, Jprim, Mprim);

									for (int int_ms1 = -S1; int_ms1 <= S1; int_ms1 = int_ms1 + 2)
									{
										for (int int_ms2 = -S2; int_ms2 <= S2; int_ms2 = int_ms2 + 2)
										{
											for (int int_ms1prim = -S1; int_ms1prim <= S1; int_ms1prim = int_ms1prim + 2)
											{
												for (int int_ms2prim = -S2; int_ms2prim <= S2; int_ms2prim = int_ms2prim + 2)
												{
													ms1 = (double)int_ms1 * 0.5;
													ms2 = (double)int_ms2 * 0.5;
													ms1prim = (double)int_ms1prim * 0.5;
													ms2prim = (double)int_ms2prim * 0.5;

													if (nonZeroSpinCG(ms1, ms2, MS, ms1prim, ms2prim, MSprim))
													{
														spinCG = SpinClebschGordanCoefficient(dS1, ms1, dS2, ms2, Stot, MS);
														spinCGprim = SpinClebschGordanCoefficient(dS1, ms1prim, dS2, ms2prim, Stotprim, MSprim);

														// diagonal part
														if (m1 == m1prim && m2 == m2prim && ms1 == ms1prim && ms2 == ms2prim)
														{
															spinRotation(i, j) += spinCG * spinCGprim * CG * CGprim * spinRotationDiagonalFunction(spinrot_strength, m1, m2, ms1, ms2);	// product base
														}

														// off-diagonal part of spin-rotation interaction: A/2 * (S_{1-} * J_{1+} + S_{2-} * J_{2-} + S_{1+} * J_{1-} + S_{2+} * J_{2-})
														if (msAndmssOffDiagonalSpinRotation(m1, m1prim, m2, m2prim, ms1, ms1prim, ms2, ms2prim))
														{
															spinRotation(i, j) += spinCG * spinCGprim * CG * CGprim * spinrot_strength * 0.5;	// product base
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return spinRotation;
}

void addToHamiltonian(Matrix& hamiltonian, Matrix& matrix)
{
	hamiltonian += matrix;
}

void buildHamiltonian(Matrix& hamiltonian, vector<int>& channels, unsigned int QN, double gjj, double gjk, double E, double B, double A, bool rotation_and_trap)
{
	if (rotation_and_trap == true)
	{
		Matrix rotation_and_trap = RotationAndTrap(channels, QN);
		hamiltonian += rotation_and_trap;
		//hamiltonian.print("Rotation and Trap: ");
	}

	if (gjj != 0)
	{
		Matrix iso_interaction = isoInteraction(channels, QN, gjj);
		hamiltonian += iso_interaction;
		//iso_interaction.print("Isotropic: ");
		//hamiltonian.print();
	}

	if (gjk != 0)
	{
		Matrix aniso_interaction = anisoInteraction(channels, QN, gjk);
		hamiltonian += aniso_interaction;
		//aniso_interaction.print("Anisotropic: ");
		//hamiltonian.print();
	}

	if (E != 0)
	{
		Matrix electric_field = electricField(channels, QN, E);
		hamiltonian += electric_field;
		//electric_field.print("Electric field: ");
		//hamiltonian.print();
	}

	if (B != 0)
	{
		Matrix magnetic_field = magneticField(channels, B);
		hamiltonian += magnetic_field;
		//magnetic_field.print("Magnetic field: ");
		//hamiltonian.print();
	}

	if (A != 0)
	{
		Matrix spin_rotation = spinRotation(channels, A);
		hamiltonian += spin_rotation;
		//spin_rotation.print("Spin-rotation: ");
		//hamiltonian.print();
	}
}

Matrix buildHamiltonianFast(vector<int>& channels, unsigned int QN, double gjj, double gjk, double E, double B, double A)
{
	int n, nprim, J, Jprim, M, Mprim, j1, j2, j1prim, j2prim, Stot=5, Stotprim=5, MS=5, MSprim=5; // to get rid of the warning about using uninitialized variable ;)
	double ms1, ms1prim, ms2, ms2prim;
	double CG, CGprim, spinCG, spinCGprim;
	unsigned int basis_size = basisSize(channels, QN);

	Matrix hamiltonian(basis_size, basis_size, 0);

	for (unsigned int i = 0; i < basis_size; i++)
	{
		n = channels[(int64_t)QN * (int64_t)i + 0];
		J = channels[(int64_t)QN * (int64_t)i + 1];
		M = channels[(int64_t)QN * (int64_t)i + 2];
		j1 = channels[(int64_t)QN * (int64_t)i + 3];
		j2 = channels[(int64_t)QN * (int64_t)i + 4];

		if (QN == QN_withspin)
		{
			Stot = channels[(int64_t)QN_withspin * (int64_t)i + 5];
			MS = channels[(int64_t)QN_withspin * (int64_t)i + 6];
		}

		// Rotation and Trap
		hamiltonian(i) += diagonalFunction(n, j1, j2);

		// Magnetic Field
		if (B != 0 && MS != 0)
		{
			hamiltonian(i) += magneticFunction(B, MS);
		}

		for (unsigned int j = 0; j < basis_size; j++)
		{
			nprim = channels[(int64_t)QN * (int64_t)j + 0];
			Jprim = channels[(int64_t)QN * (int64_t)j + 1];
			Mprim = channels[(int64_t)QN * (int64_t)j + 2];
			j1prim = channels[(int64_t)QN * (int64_t)j + 3];
			j2prim = channels[(int64_t)QN * (int64_t)j + 4];

			if (QN == QN_withspin)
			{
				Stotprim = channels[(int64_t)QN_withspin * (int64_t)j + 5];
				MSprim = channels[(int64_t)QN_withspin * (int64_t)j + 6];
			}

			// IsoInteraction
			if (gjj != 0 && differByNOnly(channels, QN, i, j))
			{
				hamiltonian(i, j) += deltaFunction(gjj, n, nprim);
			}

			// AnisoInteraction
			if (gjk != 0 && differByNAndjsOnly(channels, QN, i, j) && jsExchangeByOne(channels, QN, i, j) && momentumConserved(channels, QN, i, j))
			{
				hamiltonian(i, j) += deltaFunction(gjk, n, nprim);
			}

			// Electric Field
			if (E != 0 && sameNAndSpin(channels, QN, i, j) && onejDifferByOne(channels, QN, i, j))
			{
				for (int m1 = -j1; m1 <= j1; m1++)
				{
					for (int m2 = -j2; m2 <= j2; m2++)
					{
						for (int m1prim = -j1prim; m1prim <= j1prim; m1prim++)
						{
							for (int m2prim = -j2prim; m2prim <= j2prim; m2prim++)
							{
								if (nonZeroCG(m1, m2, M, m1prim, m2prim, Mprim) && m1 == m1prim && m2 == m2prim)
								{
									CG = ClebschGordanCoefficient(j1, m1, j2, m2, J, M);
									CGprim = ClebschGordanCoefficient(j1prim, m1prim, j2prim, m2prim, Jprim, Mprim);

									// molecules may be different
									if (j2 == j2prim)
									{
										hamiltonian(i, j) += CG * CGprim * electricFunction(mu1, E, j1, j1prim, m1);
									}

									if (j1 == j1prim)
									{
										hamiltonian(i, j) += CG * CGprim * electricFunction(mu2, E, j2, j2prim, m2);
									}

								}
							}
						}
					}
				}
			}

			// Spin-Rotation Coupling
			if (A != 0 && differByJMjSAndMsOnly(channels, i, j))
			{
				for (int m1 = -j1; m1 <= j1; m1++)
				{
					for (int m2 = -j2; m2 <= j2; m2++)
					{
						for (int m1prim = -j1prim; m1prim <= j1prim; m1prim++)
						{
							for (int m2prim = -j2prim; m2prim <= j2prim; m2prim++)
							{
								if (nonZeroCG(m1, m2, M, m1prim, m2prim, Mprim))
								{
									CG = ClebschGordanCoefficient(j1, m1, j2, m2, J, M);
									CGprim = ClebschGordanCoefficient(j1prim, m1prim, j2prim, m2prim, Jprim, Mprim);

									for (int int_ms1 = -S1; int_ms1 <= S1; int_ms1 = int_ms1 + 2)
									{
										for (int int_ms2 = -S2; int_ms2 <= S2; int_ms2 = int_ms2 + 2)
										{
											for (int int_ms1prim = -S1; int_ms1prim <= S1; int_ms1prim = int_ms1prim + 2)
											{
												for (int int_ms2prim = -S2; int_ms2prim <= S2; int_ms2prim = int_ms2prim + 2)
												{
													ms1 = (double)int_ms1 * 0.5;
													ms2 = (double)int_ms2 * 0.5;
													ms1prim = (double)int_ms1prim * 0.5;
													ms2prim = (double)int_ms2prim * 0.5;

													if (nonZeroSpinCG(ms1, ms2, MS, ms1prim, ms2prim, MSprim))
													{
														spinCG = SpinClebschGordanCoefficient(dS1, ms1, dS2, ms2, Stot, MS);
														spinCGprim = SpinClebschGordanCoefficient(dS1, ms1prim, dS2, ms2prim, Stotprim, MSprim);

														// diagonal part
														if (m1 == m1prim && m2 == m2prim && ms1 == ms1prim && ms2 == ms2prim)
														{
															hamiltonian(i) += spinCG * spinCGprim * CG * CGprim * spinRotationDiagonalFunction(A, m1, m2, ms1, ms2);	// product base
														}

														// off-diagonal part of spin-rotation interaction: A/2 * (S_{1-} * J_{1+} + S_{2-} * J_{2-} + S_{1+} * J_{1-} + S_{2+} * J_{2-})
														if (msAndmssOffDiagonalSpinRotation(m1, m1prim, m2, m2prim, ms1, ms1prim, ms2, ms2prim))
														{
															hamiltonian(i, j) += spinCG * spinCGprim * CG * CGprim * A * 0.5;	// product base
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return hamiltonian;
}

double getElectricFieldCoefficient(int j1, int j2, int J, int M, int j1prim, int j2prim, int Jprim, int Mprim, double el_field)
{
	double CG, CGprim, coefficient, sum;
	sum = 0;

	for (int m1 = -j1; m1 <= j1; m1++)
	{
		for (int m2 = -j2; m2 <= j2; m2++)
		{
			for (int m1prim = -j1prim; m1prim <= j1prim; m1prim++)
			{
				for (int m2prim = -j2prim; m2prim <= j2prim; m2prim++)
				{
					if (nonZeroCG(m1, m2, M, m1prim, m2prim, Mprim) && m1 == m1prim && m2 == m2prim)
					{
						CG = ClebschGordanCoefficient(j1, m1, j2, m2, J, M);
						CGprim = ClebschGordanCoefficient(j1prim, m1prim, j2prim, m2prim, Jprim, Mprim);

						coefficient = CG * CGprim * electricFunction(mu1, el_field, j1, j1prim, m1);
						std::cout << "m1 = " << m1 << "\t" << "m2 = " << m2 << "\t" << "m1' = " << m1prim << "\t" << "m2' = " << m2prim << " -> 1: " << coefficient << std::endl;
						sum += coefficient;

						coefficient = CG * CGprim * electricFunction(mu1, el_field, j2, j2prim, m2);
						std::cout << "m1 = " << m1 << "\t" << "m2 = " << m2 << "\t" << "m1' = " << m1prim << "\t" << "m2' = " << m2prim << " -> 2: " << coefficient << std::endl;
						sum += coefficient;
					}
				}
			}
		}
	}
	
	return sum;
}

double getInteractionCoefficient(int n, int nprim, double interaction)
{
	return deltaFunction(interaction, n, nprim);
}

double getSpinRotationCoefficient(int j1, int j2, int J, int M, int Stot, int MS, int j1prim, int j2prim, int Jprim, int Mprim, int Stotprim, int MSprim, double spinrot_strength)
{
	double CG, CGprim, spinCG, spinCGprim, coefficient, sum;
	double ms1, ms2, ms1prim, ms2prim;
	sum = 0;

	if (j1 == j1prim && j2==j2prim)
	{
		for (int m1 = -j1; m1 <= j1; m1++)
		{
			for (int m2 = -j2; m2 <= j2; m2++)
			{
				for (int m1prim = -j1prim; m1prim <= j1prim; m1prim++)
				{
					for (int m2prim = -j2prim; m2prim <= j2prim; m2prim++)
					{
						if (nonZeroCG(m1, m2, M, m1prim, m2prim, Mprim))
						{
							CG = ClebschGordanCoefficient(j1, m1, j2, m2, J, M);
							CGprim = ClebschGordanCoefficient(j1prim, m1prim, j2prim, m2prim, Jprim, Mprim);
							std::cout << "m1, m2, m1' and m2' are: " << m1 << ", " << m2 << ", " << m1prim << ", and " << m2prim << std::endl;
							std::cout << "CG and CGprim are: " << CG << " and " << CGprim << std::endl;

							for (int int_ms1 = -S1; int_ms1 <= S1; int_ms1 = int_ms1 + 2)
							{
								for (int int_ms2 = -S2; int_ms2 <= S2; int_ms2 = int_ms2 + 2)
								{
									for (int int_ms1prim = -S1; int_ms1prim <= S1; int_ms1prim = int_ms1prim + 2)
									{
										for (int int_ms2prim = -S2; int_ms2prim <= S2; int_ms2prim = int_ms2prim + 2)
										{
											ms1 = (double)int_ms1 * 0.5;
											ms2 = (double)int_ms2 * 0.5;
											ms1prim = (double)int_ms1prim * 0.5;
											ms2prim = (double)int_ms2prim * 0.5;

											if (nonZeroSpinCG(ms1, ms2, MS, ms1prim, ms2prim, MSprim))
											{
												spinCG = SpinClebschGordanCoefficient(dS1, ms1, dS2, ms2, Stot, MS);
												spinCGprim = SpinClebschGordanCoefficient(dS1, ms1prim, dS2, ms2prim, Stotprim, MSprim);
												std::cout << "spinCG and spinCGprim are: " << spinCG << " and " << spinCGprim << std::endl;

												// diagonal part
												if (m1 == m1prim && m2 == m2prim && ms1 == ms1prim && ms2 == ms2prim)
												{
													coefficient = spinCG * spinCGprim * CG * CGprim * spinRotationDiagonalFunction(spinrot_strength, m1, m2, ms1, ms2);	// product base
													std::cout << "m1 = " << m1 << "\t" << "m2 = " << m2 << "\t" << "ms1 = " << ms1 << "\t" << "ms2 = " << ms2 << "\t" << "m1' = " << m1prim << "\t" << "m2' = " << m2prim << "\t" << "ms1' = " << ms1prim << "\t" << "ms2' = " << ms2prim << "\t" << " -> 1: " << coefficient << std::endl;
													sum += coefficient;
												}

												// off-diagonal part of spin-rotation interaction: A/2 * (S_{1-} * J_{1+} + S_{2-} * J_{2-} + S_{1+} * J_{1-} + S_{2+} * J_{2-})
												if (msAndmssOffDiagonalSpinRotation(m1, m1prim, m2, m2prim, ms1, ms1prim, ms2, ms2prim))
												{
													coefficient = spinCG * spinCGprim * CG * CGprim * spinrot_strength * 0.5;	// product base
													std::cout << "m1 = " << m1 << "\t" << "m2 = " << m2 << "\t" << "ms1 = " << ms1 << "\t" << "ms2 = " << ms2 << "\t" << "m1' = " << m1prim << "\t" << "m2' = " << m2prim << "\t" << "ms1' = " << ms1prim << "\t" << "ms2' = " << ms2prim << "\t" << " -> 1: " << coefficient << std::endl;
													sum += coefficient;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	else
	{
		sum = 0;
	}

	return sum;
}