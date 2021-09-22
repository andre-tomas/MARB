clear all;
syms(sym('w', [2 3]));
syms(sym('d',[1 3]));

H = [0, 0, w1_1, w1_2, w1_3;
      0, 0, w2_1, w2_2, w2_3;
      conj(w1_1), conj(w2_1), 0, 0, 0;
      conj(w1_2), conj(w2_2), 0, 0, 0;
      conj(w1_3), conj(w2_3), 0, 0, 0];
  
      
  
  H_e = simplify(eig(H));

  
  T = [1, 0, 0, 0, 0;
       0, 1, 0, 0, 0;
       0, 0, w1_1, w2_1, d1;
       0, 0, w1_2, w2_2, d2;
       0, 0, w1_3, w2_3, d3];
   Td = ctranspose(T);
   
   
   H_d = simplify(Td*H*T);
   
   old_Hd = H_d;
   
   
   syms(sym('k', [1 3]));
   
   H_d = [0, 0, k1, k3;
          0, 0, k3, k2;
          conj(k1), conj(k3), 0, 0; 
          conj(k3), conj(k2), 0, 0];
      
      simplify(eig(H_d)/sqrt(2))
   
   