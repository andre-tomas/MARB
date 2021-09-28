clear all;
syms(sym('w', [2 3]));
%syms(sym('d',[1 3]));
syms K
syms(sym('a', [1 2]))



%H = [0, 0, w1_1, w1_2, w1_3;
 %     0, 0, w2_1, w2_2, w2_3;
  %    conj(w1_1), conj(w2_1), 0, 0, 0;
   %   conj(w1_2), conj(w2_2), 0, 0, 0;
    %  conj(w1_3), conj(w2_3), 0, 0, 0];

    H = [0, 0, w1_1, w1_2, a1, 0;
         0, 0, w2_1, w2_2, 0, a2;
         conj(w1_1), conj(w2_1), 0, 0, 0, 0;
         conj(w1_2), conj(w2_2), 0, 0, 0, 0;
         conj(a1), 0, 0, 0, 0, 0;
         0, conj(a2), 0, 0, 0, 0];
      
  
  H_e = simplify(eig(H));

  
  
    
  %T = [1, 0, 0, 0, 0, 0;
     %  0, 1, 0, 0, 0, 0;
     %  0, 0, 0, -conj(d2), d1;
      % 0, 0, 0, conj(d1), d2;
      % 0, 0, 1, 0, 0];
   %Td = ctranspose(T);
   
  d1 = (1/a1)*(w2_2*w1_1 - w1_2*w2_1);
  d2 = -d1*a1/a2;
  
  T = [1, 0, 0, 0, 0, 0;
       0, 1, 0, 0, 0, 0;
       0, 0, -w1_1, -w2_1, -w2_2, -w1_2;
       0, 0, -w1_2, -w2_2, w2_1, w1_1;
       0, 0, -a1, 0, d1, 0;
       0, 0, 0, -a2, 0, d2];
   Td = ctranspose(T);
   
   
   H_d = simplify(simplify(Td*H*T));
   
   old_Hd = H_d
   
   