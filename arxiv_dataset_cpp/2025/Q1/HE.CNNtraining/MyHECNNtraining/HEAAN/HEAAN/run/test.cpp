/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#include "../src/HEAAN.h"
/**
  * This file is for test HEAAN library
  * You can find more in src/TestScheme.h
  * "./TestHEAAN Encrypt" will run Encrypt Test
  * There are Encrypt, EncryptSingle, Add, Mult, iMult, RotateFast, Conjugate Tests
  */

/* To Consist With the IDASH2017, There Should Be Only One main() Function. */
/*              The error is : multiple definition of `main';               */
int MAIN() {

    	long logp = 35;
    	long logq = logp + 10; //< suppose the input ciphertext of bootstrapping has logq = logp + 10
    	long logSlots = 9; //< larger logn will make bootstrapping tech much slower
    	long logT = 4; //< this means that we use Taylor approximation in [-1/T,1/T] with double angle fomula


        //TestScheme::testBootstrap(logq, logp, logSlots, logT) {
	cout << "!!! START TEST BOOTSTRAP !!!" << endl;

	srand(time(NULL));
	SetNumThreads(8);
	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	timeutils.start("Key generating");
	scheme.addBootKey(secretKey, logSlots, logq + 4);
	timeutils.stop("Key generated");

	long slots = (1 << logSlots);
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);

	Ciphertext cipher;
	scheme.encrypt(cipher, mvec, slots, logp, logq);

	cout << endl;
	cout << "scheme.bootstrapAndEqual(cipher, logq, logQ, logT);" << endl << endl;
	cout << "cipher logp before: " << cipher.logp << endl;
	cout << "cipher logq before: " << cipher.logq << endl;

	timeutils.start("scheme.bootstrapAndEqual");
	scheme.bootstrapAndEqual(cipher, logq, logQ, logT);
	timeutils.stop("scheme.bootstrapAndEqual");



	complex<double>* dvec = scheme.decrypt(secretKey, cipher);

	StringUtils::compare(mvec, dvec, slots, "boot");

	cout << "cipher logp after : " << cipher.logp << endl;
	cout << "cipher logq after : " << cipher.logq << endl;




	cout << "!!! END TEST BOOTSRTAP !!!" << endl;



	return 0;
}
