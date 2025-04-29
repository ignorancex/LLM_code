#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <ostream>
#include <math.h>

/**
 * Version: August 2024
 */

/**
 * This class creates a mathematical Vector, with components x, y and z (3D system)
 */
class Vector {
	public:
		float x, y, z;

        /**
         * Basic constructor
         * @param v_x Is the x component of the vector (default=0)
         * @param v_y Is the y component of the vector (default=0)
         * @param v_z Is the z component of the vector (default=0)
         */
		Vector(float v_x= 0, float v_y= 0, float v_z= 0) {
			x= v_x;
			y= v_y;
			z= v_z;
		}

        /**
         * Constructor for assigment
         * @param v Other vector already defined
         */
		Vector(const Vector& v): x(v.x), y(v.y), z(v.z) {}

		/**
         * @return module of the vector
         */
		float magnitude() const {
			return sqrt(x*x+y*y+z*z);
		}

		/**
         * To compare two vectors in magnitude. Otherwise, use ==
         * @param v Other vector to compare
         * @return true if both vectors have the same magnitude (be careful with the precision managed)
         */
		bool equalMagnitud(Vector& v) {
			return magnitude()==v.magnitude();
		}

		/**
         * OVERLOAD of operator + for two vectors
         * @param v Other vector to sum
         * @return The sum of each component in a new *Vector
         */
		Vector* operator +(const Vector& v) {
			return new Vector(x+v.x, y+v.y, z+v.z);
		}

		/**
         * OVERLOAD of operator - for two vectors
         * @param v Other vector to substract
         * @return The substraction of each component in a new *Vector
         */
		Vector* operator -(const Vector& v) {
			return new Vector(x-v.x, y-v.y, z-v.z);
		}

		/**
         * OVERLOAD of operator * for two vectors
         * @param v Other vector to operate
         * @return The scalar product of both vectors
         */
		float operator *(const Vector& v) {
			return x*v.x + y*v.y + z*v.z;
		}

		/**
         * OVERLOAD of operator * for one vector and a scalar
         * @param k A constant (scalar)
         * @return The product of each component with the scalar k in a new *Vector
         */
		Vector* operator *(const float k) {
			return new Vector(x*k, y*k, z*k);
		}

		/**
         * OVERLOAD of operator / for one vector and a scalar
         * @param k A constant (scalar)
         * @return The division of each component with the scalar k in a new *Vector
         */
		Vector* operator /(const float k) {
			return new Vector(x/k, y/k, z/k);
		}

        /**
         * OVERLOAD of operator % for two vectors
         * @param v Other vector to operate
         * @return The cross (vector) product of both vectors in a new *Vector
         */
		Vector* operator %(const Vector& v) {
			return new Vector(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x);
		}

		/**
		 * OVERLOAD of operator == for two vectors
         * To compare two vectors in component. For comparing magnitud use equalsMagnitud()
         * @param v Other vector to compare
         * @return true if both vectors have the same components in x, y, and z (be careful with the precision managed)
         */
		bool operator ==(const Vector& v) {
			return x==v.x && y==v.y && z==v.z;
		}

		/**
		 * OVERLOAD of operator != for two vectors
         * To compare two vectors in component. For comparing magnitud use !equalsMagnitud()
         * @param v Other vector to compare
         * @return true if both vectors DO NOT have the same components in x, y, and z
         */
		bool operator !=(const Vector& v) {
			return !(*this==v);
		}

		/**
		 * OVERLOAD of operator < for two vectors
         * To compare two vectors in magnitud
         * @param v Other vector to compare
         * @return true if this vector has a minor magnitud than v
         */
		bool operator <(const Vector& v) {
			return magnitude()<v.magnitude();
		}

		/**
		 * OVERLOAD of operator > for two vectors
         * To compare two vectors in magnitud
         * @param v Other vector to compare
         * @return true if this vector has a major magnitud than v
         */
		bool operator >(const Vector& v) {
			return magnitude()>v.magnitude();
		}

		/**
		 * OVERLOAD of operator <= for two vectors
         * To compare two vectors in magnitud
         * @param v Other vector to compare
         * @return true if this vector has a minor or equal magnitud than v
         */
		bool operator <=(const Vector& v) {
			return magnitude()<=v.magnitude();
		}

		/**
		 * OVERLOAD of operator >= for two vectors
         * To compare two vectors in magnitud
         * @param v Other vector to compare
         * @return true if this vector has a major or equal magnitud than v
         */
		bool operator >=(const Vector& v) {
			return magnitude()>=v.magnitude();
		}

};

/**
 * OVERLOAD of operator << for a Vector
 * Writes the components of the vector in the format: -->(x,y,z)
 * @param o Ostream to write in
 * @param v Vector to write
 * @return Ostream with the writed string
 */
std::ostream& operator <<(std::ostream& o, Vector& v) {
    o << "-->(" << v.x << "," << v.y << "," << v.z << ")";
    return o;
}

/**
 * OVERLOAD of operator << for a *Vector
 * Writes the components of the vector in the format: -->(x,y,z)
 * @param o Ostream to write in
 * @param v Vector to write
 * @return Ostream with the writed string
 */
std::ostream& operator <<(std::ostream& o, Vector* v) {
    o << "-->(" << v->x << "," << v->y << "," << v->z << ")";
    return o;
}

struct Tetrahedron {
    Vector* H1;
    Vector* H2;
    Vector* L1;
    Vector* L2;
};

/**
 * Calculates the distance between two coordinates (vectors) given the bounds to use periodic boundary conditions
 * @param c1 One of the coordinates
 * @param c2 The other coordinate
 * @param bounds The coordinate of the last point, so the components are the width, height and length
 * @return The distance between the two coordinates
 */
float dist(Vector* c1, Vector* c2, Vector* bounds) {
    Vector* dif= *c1-*c2;
    Vector* b_1= *bounds/2;
    Vector* b_2= *bounds/(-2);

    if(dif->x > b_1->x) dif->x-= bounds->x;
    if(dif->x < b_2->x) dif->x+= bounds->x;
    if(dif->y > b_1->y) dif->y-= bounds->y;
    if(dif->y < b_2->y) dif->y+= bounds->y;
    if(dif->z > b_1->z) dif->z-= bounds->z;
    if(dif->z < b_2->z) dif->z+= bounds->z;

    float output= dif->magnitude();

    delete(dif);
    delete(b_1);
    delete(b_2);

    return output;
}

/**
 * Calculates the determinant of a 3x3 matrix with the Sarrus Rule
 * @param matrix A 3x3 float array
 * @return The matrix's determinant
 */
float determinant_3x3(float matrix[3][3]) {
    return matrix[0][0]*matrix[1][1]*matrix[2][2] + matrix[0][1]*matrix[1][2]*matrix[2][0] + matrix[0][2]*matrix[1][0]*matrix[2][1]
         - matrix[0][2]*matrix[1][1]*matrix[2][0] - matrix[0][0]*matrix[1][2]*matrix[2][1] - matrix[0][1]*matrix[1][0]*matrix[2][2];
}

/**
 * Calculates the x,y,z values (a Vector) obteined from the Cramer's Rule
 * @param matrix A 3x3 float array of the base matrix
 * @param m_independents A float array of 3 values corresponding to the independt values
 * @return The Vector with the x, y and z values obtained. (The Vector must be removed with "delete()")
 */
Vector* CramersRule(float matrix[3][3], float m_independents[3]) {
        float A= determinant_3x3(matrix);
        float Ax_matrix[3][3]= {
            { m_independents[0] , matrix[0][1] , matrix[0][2] },
            { m_independents[1] , matrix[1][1] , matrix[1][2] },
            { m_independents[2] , matrix[2][1] , matrix[2][2] }
        };
        float Ax= determinant_3x3(Ax_matrix);
        float Ay_matrix[3][3]= {
            { matrix[0][0] , m_independents[0] , matrix[0][2] },
            { matrix[1][0] , m_independents[1] , matrix[1][2] },
            { matrix[2][0] , m_independents[2] , matrix[2][2] }
        };
        float Ay= determinant_3x3(Ay_matrix);
        float Az_matrix[3][3]= {
            { matrix[0][0] , matrix[0][1] , m_independents[0] },
            { matrix[1][0] , matrix[1][1] , m_independents[1] },
            { matrix[2][0] , matrix[2][1] , m_independents[2] }
        };
        float Az= determinant_3x3(Az_matrix);
        return new Vector(Ax/A, Ay/A, Az/A);
}

Vector** checkPBC(Vector* O, Vector* H1, Vector* H2, Vector* bounds) {
    Vector** output= new Vector*[3];

    output[0]= new Vector(O->x,O->y,O->z);
    Vector* H_i[2]= {H1, H2};
    for(int i= 0; i < 2; i++) {
        Vector* dif= *H_i[i]-*O;
        Vector* b_1= *bounds/2;
        Vector* b_2= *bounds/(-2);

        if(dif->x > b_1->x) dif->x-= bounds->x;
        if(dif->x < b_2->x) dif->x+= bounds->x;
        if(dif->y > b_1->y) dif->y-= bounds->y;
        if(dif->y < b_2->y) dif->y+= bounds->y;
        if(dif->z > b_1->z) dif->z-= bounds->z;
        if(dif->z < b_2->z) dif->z+= bounds->z;

        output[i+1]= *O+*dif;

        delete(dif);
        delete(b_1);
        delete(b_2);
    }
    return output;
}

/**
 * Calculates the angle that 3 points form in a 3D system
 * @param c1 One of the point in the edges
 * @param c2 The center point
 * @param c3 The other point in an edge
 * @param bounds The coordinate of the last point, so the components are the width, height and length
 * @return The angle in radians formed by the 3 points
 */
float getAngle(Vector* c1, Vector* c2, Vector* c3, Vector* bounds) {
    float a= dist(c1,c3,bounds); //Opposite to the angle
    float b= dist(c1,c2,bounds);
    float c= dist(c2,c3,bounds);
    //Law of cosines
    return abs(acos((pow(b,2)+pow(c,2)-pow(a,2))/(2*b*c)));
}

Tetrahedron getPerfectTetrahedron(Vector* O_real, Vector* H1_real, Vector* H2_real, Vector* bounds) {
    const float R= 1; //Anstrongs between O and perfect vertices
    const float theta= acos(-1./3.)/2; //Perfect angle for tetrahedron
    float phy= getAngle(H1_real,O_real,H2_real,bounds)/2; //I need the angle between OH and b

    Vector** pbc_vectors= checkPBC(O_real, H1_real, H2_real, bounds);
    Vector* O= pbc_vectors[0];
    Vector* H1= pbc_vectors[1];
    Vector* H2= pbc_vectors[2];
    delete(pbc_vectors);

    //I step
    Vector* OH1= (*H1-*O);
    Vector* OH2= (*H2-*O);
    Vector* OH1_norm= *OH1*(1./OH1->magnitude());
    Vector* OH2_norm= *OH2*(1./OH2->magnitude());

    //II step
    Vector* b= *OH1_norm + *OH2_norm;
    delete(OH1_norm);
    delete(OH2_norm);

    //III step
    Vector* nu_OH_sin_norm= *OH1 % *OH2;
    Vector* nu_OH= *nu_OH_sin_norm*(1./nu_OH_sin_norm->magnitude());
    delete(nu_OH_sin_norm);

    //IV step
    Vector* hidrogenos[2]= {OH1, OH2};
    Vector* h[2];
    for(int i= 0; i < 2; i++) {
        Vector* OHi= hidrogenos[i];
        float k1= R*b->magnitude()*cos(theta);
        float k2= R*OHi->magnitude()*cos(theta-phy);

        float A_matriz[3][3]= {
            {   b->x   ,   b->y   ,   b->z   },
            {  OHi->x  ,  OHi->y  ,  OHi->z  },
            { nu_OH->x , nu_OH->y , nu_OH->z }
        };
        delete(OHi); //delete OH1,2
        float m_independientes[3]= { k1, k2, 0. };

        Vector* oh= CramersRule(A_matriz, m_independientes);
        h[i]= *oh + *O;
        delete(oh);
    }
    delete(b);

    //V step
    Vector* suma_h= *h[0]+*h[1];
    Vector* m_H= *suma_h*(0.5);
    delete(suma_h);
    Vector* mH_H= *h[0] - *m_H;
    float delta= mH_H->magnitude();
    delete(mH_H);

    //VI step
    Vector* O_2= *O*2.;
    Vector* m_L= *O_2 - *m_H;
    delete(O_2);
    delete(m_H);

    //VII step
    Vector* mL_L1= *nu_OH*delta;
    Vector* mL_L2= *nu_OH*(-delta);
    delete(nu_OH);
    Vector* L1= *mL_L1 + *m_L;
    Vector* L2= *mL_L2 + *m_L;
    delete(mL_L1);
    delete(mL_L2);
    delete(m_L);

    Tetrahedron output;
    output.H1= h[0];
    output.H2= h[1];
    output.L1= L1;
    output.L2= L2;

    delete(O);
    delete(H1);
    delete(H2);

    return output;
}


struct Atom {
    int serial;
    std::string name;
    std::string resName;
    char chainID;
    int resSeq;
    float x, y, z;
    float occupancy;
    float tempFactor;
};

struct Box {
    float x, y, z;
};

struct MoleculeData {
    std::string name;
    float charge;
    float sigma;
    float epsilon;
};

struct Molecule {
    std::string resName;
    std::vector<Atom> atoms;
};

float distanceBetween(Atom a1, Atom a2, Box b) {
    Box dif;
    dif.x= a1.x-a2.x;
    dif.y= a1.y-a2.y;
    dif.z= a1.z-a2.z;
    Box b_1;
    b_1.x= b.x/2;
    b_1.y= b.y/2;
    b_1.z= b.z/2;
    Box b_2;
    b_2.x= -b.x/2;
    b_2.y= -b.y/2;
    b_2.z= -b.z/2;

    if(dif.x > b_1.x) dif.x-= b.x;
    if(dif.x < b_2.x) dif.x+= b.x;
    if(dif.y > b_1.y) dif.y-= b.y;
    if(dif.y < b_2.y) dif.y+= b.y;
    if(dif.z > b_1.z) dif.z-= b.z;
    if(dif.z < b_2.z) dif.z+= b.z;

    return sqrt(dif.x*dif.x+dif.y*dif.y+dif.z*dif.z);
}

float distanceBetween(Molecule m, Atom a_ref, Box b) {
    return distanceBetween(m.atoms[0],a_ref,b);
}

float distanceBetween(Molecule m1, Molecule m2, Box b) {
    return distanceBetween(m1.atoms[0],m2.atoms[0],b);
}

float potentialBetween(const Molecule& w, const Atom& a, const Box& b, const std::unordered_map<std::string, MoleculeData>& md) {
    float totalPotential = 0.0;

    if (md.find(a.name) == md.end()) return 0.0; // Si no hay datos de la molécula, retornar 0
    const MoleculeData& a_md = md.at(a.name);

    for (const auto& atom : w.atoms) {
        if (md.find(atom.name) == md.end()) continue; // Si no hay datos del átomo en la molécula, continuar
        const MoleculeData& w_md = md.at(atom.name);

        float r = distanceBetween(a, atom, b);
        float s = 0.5*(w_md.sigma + a_md.sigma);
        float e = sqrt(w_md.epsilon * a_md.epsilon);

        float ljPotential = 4 * e * (std::pow(s / r, 12) - std::pow(s / r, 6));
        float coulombPotential = 1389.35458*(a_md.charge * w_md.charge) / r;

        totalPotential += ljPotential + coulombPotential;
    }

    return totalPotential;
}

float potentialBetween(const Molecule& w1, const Molecule& w2, const Box& b, const std::unordered_map<std::string, MoleculeData>& md) {
    float totalPotential = 0.0;

    for (const auto& atom : w2.atoms) {
        totalPotential += potentialBetween(w1,atom,b,md);
    }

    return totalPotential;
}

float computeV4S(Molecule m, std::vector<Molecule> w_molecs, std::vector<Molecule> nw_molecs, Box box, const std::unordered_map<std::string, MoleculeData>& molec_data) {
    const float R_CUT_OFF= 5.;

    Vector* ox= new Vector(m.atoms[0].x,m.atoms[0].y,m.atoms[0].z);
    Vector* h1= new Vector(m.atoms[1].x,m.atoms[1].y,m.atoms[1].z);
    Vector* h2= new Vector(m.atoms[2].x,m.atoms[2].y,m.atoms[2].z);
    Vector* bounds= new Vector(box.x,box.y,box.z);
    Tetrahedron t= getPerfectTetrahedron(ox, h1, h2, bounds);

    float* sum_per_site= new float[4];
    for(int i= 0; i < 4; i++)
        sum_per_site[i]= 0.0;

    Vector* sites[4]= {t.H1, t.H2, t.L1, t.L2};

    for (Molecule molecule : w_molecs) {
        if(m.atoms[0].serial == molecule.atoms[0].serial) continue;
        if(distanceBetween(m,molecule,box) > R_CUT_OFF+1.1) continue;

        Vector* position= new Vector(molecule.atoms[0].x,molecule.atoms[0].y,molecule.atoms[0].z);
        int i_close= 0;
        float d_close= dist(sites[0],position,bounds);

        for(int i= 1; i < 4; i++) {
            float d_new= dist(sites[i],position,bounds);
            if(d_close > d_new) {
                i_close= i;
                d_close= d_new;
            }
        }

        delete(position);

        if(d_close <= R_CUT_OFF)
            sum_per_site[i_close]+= potentialBetween(m,molecule,box,molec_data);
    }

    for (Molecule molecule : nw_molecs) {
        for (Atom atom: molecule.atoms) {
            if(distanceBetween(m,atom,box) > R_CUT_OFF+1.1) continue;

            Vector* position= new Vector(atom.x,atom.y,atom.z);
            int i_close= 0;
            float d_close= dist(sites[0],position,bounds);

            for(int i= 1; i < 4; i++) {
                float d_new= dist(sites[i],position,bounds);
                if(d_close > d_new) {
                    i_close= i;
                    d_close= d_new;
                }
            }

            delete(position);

            if(d_close <= R_CUT_OFF)
                sum_per_site[i_close]+= potentialBetween(m,atom,box,molec_data);
        }
    }

    delete(bounds);
    for(int i= 0; i < 4; i++)
        delete(sites[i]);

    float v4s= sum_per_site[0];
    for(int i= 1; i < 4; i++)
        if(sum_per_site[i] > v4s)
            v4s= sum_per_site[i];
    delete(sum_per_site);
    return v4s;
}

std::string floatToStringWithPadding(float value, int width) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value;
    std::string str = oss.str();

    if (str.length() < width) {
        str.append(width - str.length(), ' ');
    }

    return str;
}

std::string trim(const std::string& str) {
    std::string result = str;
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    return result;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream inputFile("files.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file files.txt" << std::endl;
        return 1;
    }

    std::string line;
    std::string file1, file2;
    int count = 0;
    float radii_study_minimum= 0.0;
    float radii_study_maximum= 3.5;

    // Read lines from the input file
    while (std::getline(inputFile, line)) {
        // Ignore lines that start with #
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Store the file names
        if (count == 0) {
            file1 = line;
            count++;
        } else if (count == 1) {
            file2 = line;
            count++;
        } else if (count == 2) {
            radii_study_minimum= std::stof(line);
            count++;
            std::cout << "Molecules will be studied at a minimum distance of: " << radii_study_minimum << std::endl;
        } else if (count == 3) {
            radii_study_maximum= std::stof(line);
            std::cout << "Molecules will be studied at a maximum distance of: " << radii_study_maximum << std::endl;
            break;
        }
    }

    inputFile.close();

    // Check if two file names were found
    if (file1.empty() || file2.empty()) {
        std::cerr << "Not enough file names found in files.txt" << std::endl;
        return 1;
    }

    std::unordered_map<std::string, MoleculeData> moleculeData;

    // Open and read the second file (Molecule data)
    std::ifstream fileStream2(file2);
    if (!fileStream2.is_open()) {
        std::cerr << "Error opening file " << file2 << std::endl;
        return 1;
    }

    while (std::getline(fileStream2, line)) {
        // Ignore empty lines and lines that start with #
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        MoleculeData molecule;
        iss >> molecule.name >> molecule.charge >> molecule.sigma >> molecule.epsilon;
        moleculeData[molecule.name] = molecule;
    }

    fileStream2.close();

    // Variable to store the residue name for water
    std::string waterResName = "WAT";

    // Open and read the PDB file (first file)
    std::ifstream pdbFile(file1);
    if (!pdbFile.is_open()) {
        std::cerr << "Error opening file " << file1 << std::endl;
        return 1;
    }

    Box box;
    std::map<int, Molecule> molecules;
    std::vector<std::string> pdbLines;

    int res_seq_old= 0;
    int i_molec= 0;

    while (std::getline(pdbFile, line)) {
        pdbLines.push_back(line);
        if (line.substr(0, 6) == "CRYST1") {
            // Parse box size information
            box.x = std::stof(line.substr(6, 9));
            box.y = std::stof(line.substr(15, 9));
            box.z = std::stof(line.substr(24, 9));
        } else if (line.substr(0, 4) == "ATOM") {
            int res_seq_new= std::stoi(line.substr(22, 4));
            if(res_seq_new != res_seq_old) {
                res_seq_old= res_seq_new;
                i_molec++;
            }
            // Parse atom information
            Atom atom;
            atom.serial = std::stoi(line.substr(6, 5));
            atom.name = trim(line.substr(11, 5));
            atom.resName = trim(line.substr(16, 5));
            atom.chainID = line[21];
            atom.resSeq = i_molec;
            atom.x = std::stof(line.substr(30, 8));
            atom.y = std::stof(line.substr(38, 8));
            atom.z = std::stof(line.substr(46, 8));
            atom.occupancy = std::stof(line.substr(54, 6));
            atom.tempFactor = std::stof(line.substr(60, 6));

            // Group atoms by resSeq
            molecules[i_molec-1].resName = atom.resName;
            molecules[i_molec-1].atoms.push_back(atom);
        }
    }

    pdbFile.close();

    // Separate molecules into water and non-water groups
    std::vector<Molecule> waterMolecules;
    std::vector<Molecule> nonWaterMolecules;

    for (const auto& pair : molecules) {
        if (pair.second.resName == waterResName) {
            waterMolecules.push_back(pair.second);
        } else {
            nonWaterMolecules.push_back(pair.second);
        }
    }

    // Print the box size information and molecules information
    std::cout << "Box size: " << box.x << " " << box.y << " " << box.z << std::endl;
    std::cout << "Water molecules: " << waterMolecules.size() << "\tNon water molecules: " << nonWaterMolecules.size() << std::endl;

    int number_of_hidrophylic_molecules= 0;
    int number_of_hidrophofic_molecules= 0;

    for (int i_molecule= 0; i_molecule < waterMolecules.size(); i_molecule++) {
        bool compute_v4s= false;
        for (Molecule nwm : nonWaterMolecules) {
            for (Atom a_nwm : nwm.atoms) {
                float distance= distanceBetween(waterMolecules[i_molecule],a_nwm,box);
                if(distance <= radii_study_maximum && distance >= radii_study_minimum) {
                    compute_v4s= true;
                    break;
                }
            }
        }
        if(!compute_v4s) continue;
        float v4s = computeV4S(waterMolecules[i_molecule], waterMolecules, nonWaterMolecules, box, moleculeData);
        for (int i_atom= 0; i_atom < waterMolecules[i_molecule].atoms.size(); i_atom++) {
            waterMolecules[i_molecule].atoms[i_atom].tempFactor = v4s;
        }

        if(v4s < -12)
            number_of_hidrophylic_molecules++;
        else
            number_of_hidrophofic_molecules++;
    }

    std::cout << "Number of molecules with V4S < -12: " << number_of_hidrophylic_molecules << std::endl;
    std::cout << "Number of molecules with V4S > -12: " << number_of_hidrophofic_molecules << std::endl;
    std::cout << "Writing output file: modified_" + file1 << std::endl;

    // Escribir el archivo PDB actualizado
    std::ofstream updatedPdbFile("modified_" + file1);
    if (!updatedPdbFile.is_open()) {
        std::cerr << "Error opening file to write updated PDB" << std::endl;
        return 1;
    }

    int i_line= 0;
    for (const auto& line : pdbLines) {
        if (line.substr(0, 4) == "ATOM") {
            break;
        }
        updatedPdbFile << line << std::endl;
        i_line++;
    }

    for (int i= 0; i < nonWaterMolecules.size(); i++) {
        for(int j= 0; j < nonWaterMolecules[i].atoms.size(); j++) {
            updatedPdbFile << pdbLines[i_line] << std::endl;
            i_line++;
        }
    }

    for (Molecule molecule : waterMolecules) {
        for (Atom atom : molecule.atoms) {
            // Reemplazar tempFactor en la línea original
            std::string line= pdbLines[i_line];
            std::string updatedLine = line.substr(0, 62) + floatToStringWithPadding(atom.tempFactor,8) + line.substr(70);
            updatedPdbFile << updatedLine << std::endl;
            i_line++;
        }
    }

    updatedPdbFile.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;


    return 0;
}
