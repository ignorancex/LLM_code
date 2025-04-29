#include "Configuration.h"
#include "Tetrahedron.h"

const int N_CONF= 10000;

float* getInteractions(Configuration* c, const int ID) {
    const float R_CUT_OFF= 5.;
    Water* molecule= (Water*) c->getMolec(ID);
    Vector* o= molecule->getAtom_O()->getPosition();
    Vector* h1= molecule->getAtom_H1()->getPosition();
    Vector* h2= molecule->getAtom_H2()->getPosition();
    Tetrahedron t= getPerfectTetrahedron(o, h1, h2, getAngle(h1,o,h2,c->getBounds()), c->getBounds());

    float* sum_per_point= new float[4];
    for(int i= 0; i < 4; i++)
        sum_per_point[i]= 0;

    Vector* tetrah[4]= {t.H1, t.H2, t.L1, t.L2};

    for(int j= 1; j <= c->getNMolec(); j++) {
        if(j==ID) continue;
        if(c->getMolec(ID)->distanceTo(c->getMolec(j), c->getBounds()) > R_CUT_OFF+1) continue;

        int i_close= 0;
        float d_close= dist(tetrah[0],c->getMolec(j)->getPosition(),c->getBounds());
        for(int i= 1; i < 4; i++) {
            float new_d= dist(tetrah[i],c->getMolec(j)->getPosition(),c->getBounds());
            if(d_close > new_d) {
                i_close= i;
                d_close= new_d;
            }
        }
        if(d_close <= R_CUT_OFF)
            sum_per_point[i_close]+= molecule->potentialWith((Water*)c->getMolec(j),c->getBounds());
    }

    for(int i= 0; i < 4; i++)
        delete(tetrah[i]);

    sort(sum_per_point,4);
    return sum_per_point;
}

int main() {
    Configuration* c;

    for(int conf= 0; conf <= N_CONF; conf++) {
        c= new Configuration(getDir(conf),WMODEL_TIP3P);

        for(int m= 1; m <= c->getNMolec(); m++) {
            float* interaction= getInteractions(c, m);
            cout << m << "\t";
            for(int i= 0; i < 4; i++) {
                cout << interaction[i] << "\t";
            }
            cout << endl;
            delete(interaction);
        }
        delete(c);
    }

    return 0;
}
