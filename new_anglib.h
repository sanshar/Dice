/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma, Garnet K.-L. Chan and Roberto Olivares-Amaya
*/

double nine_j(int na, int nb, int nc, int nd, int ne, int nf, int ng, int nh, int ni);

double six_j(int j1, int j2, int j3, int l1, int l2, int l3);

double j6_delta(double a, double b, double c);

double square_six(double a, double b, double c, double d, double e, double f);

double three_j(int j1, int j2, int j3, int m1, int m2, int m3);

double clebsch(int j1, int m1, int j2, int m2, int j3, int m3);

double facto(double n);

int mone(double x);

int get_cast(double x);

double fbinom(double n, double r);
