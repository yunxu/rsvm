# include <stdio.h>
# include <stdlib.h>
# include <ctype.h>
# include <math.h>
# include <string.h>
#include <time.h>
#define MAX_LINE_LEN 19999
#define NM_MAX  9999 
#define TINY 1.0e-14 // A small number.

typedef struct sampe_doc {
  long    ind;      //index number
  char    nam[66];  // sample name
  int     p;        // 1 for positive sample; -1 for negative sample;
  double *weight;
} DOC;

typedef struct model {
  long    sv_num;	
  long    at_upper_bound;
  double  b;
  DOC     **supvec;
  double  *alpha;
  long    *index;       /* index from docnum to position in model */
  long    totwords;     /* number of features */
  long    totdoc;       /* number of training documents */
} MODEL;

void svm_learn(DOC *p_d, DOC *n_d);
void n_svm(double *A,  double *w, double *gamma,DOC *p_d, DOC *n_d);
void pre_process_string(char *str);
void random_1_nn(int nn, int *arr);
double vector_norm(int nn, double *xx);
