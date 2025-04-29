typedef struct {
  int x[5];
} ascon_state_t;

void sbox(ascon_state_t state) {
     int t0, t1, t2, t3, t4;
     state.x[0] ^= state.x[4];    
     state.x[4] ^= state.x[3];    
     state.x[2] ^= state.x[1];
     t0  = state.x[0];    
     t1  = state.x[1];    
     t2  = state.x[2];    
     t3  = state.x[3];    
     t4  = state.x[4];
     t0 =~ t0;    
     t1 =~ t1;    
     t2 =~ t2;    
     t3 =~ t3;    
     t4 =~ t4;
     t0 &= state.x[1];   
     t1 &= state.x[2];    
     t2 &= state.x[3];    
     t3 &= state.x[4];    
     t4 &= state.x[0];
     state.x[0] ^= t1;   
     state.x[1] ^= t2;    
     state.x[2] ^= t3;    
     state.x[3] ^= t4;    
     state.x[4] ^= t0;
     state.x[1] ^= state.x[0];
     state.x[0] ^= state.x[4]; 
     state.x[3] ^= state.x[2]; 
     state.x[2] =~ state.x[2];

     return;
}

