/***********************************************************************/
/*                                                                     */
/*   rdsvm_learn.cpp                                                   */
/*                                                                     */
/*   Command line interface to the learning module of the              */
/*   reduce Support Vector Machine.                                    */
/*                                                                     */
/*   Author: Yun Xu                                                    */
/*   Date: Apri 24, 2009                                               */
/*                                                                     */
/*   Copyright (c) 2009  University of Illinois - All rights reserved  */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/
#include "rdsvm_learn.h"
#pragma warning(disable : 4996)

char pos_doc_file[99],neg_doc_file[99];     // file with training examples */
char model_file[99];         /* file for resulting classifier */
char tmp_line[MAX_LINE_LEN];
int num_tot, num_p, num_n, num_var;
int num_SVs_p, num_SVs_n, *SVs_p, *SVs_n, num_SVs_tot;
int auto_scal, visible;
double nu,J_nu, p_rate,n_rate, gamma_val,kernel_t,stop_criteria;
bool debug;
FILE *fpr, *fpw, *fp1;

#ifdef WIN32
	extern "C"  int DGESV(int *, int *, double *,  int *, int *, double *, int *, int *);
	extern "C"  int DGESVD(char *, char *, int *, int *, double *, int *, double *, double *, int *,double *, int *, double *, int *, int *);
#else
	extern "C"  int dgesv_(int *, int *, double *,  int *, int *, double *, int *, int *);
	extern "C"  int dgesvd_(char *, char *, int *, int *, double *, int *, double *, double *, int *,double *, int *, double *, int *, int *);
#endif

char pdb_nam_ar[9999][8];
int  residu_num_ar[9999];
int  TOTAL;

int main (int argc, char* argv[]){

    int i, j, k, kk, mm, pos, *arr_t;
    double tt, xx;
    double *av, *sd;
    char str1[99], str2[99]; //,temp[99];
    DOC *docs_p, *docs_n;  /* training examples */;
    DOC *docs_t;


//    rate = 0.25;
//    nu = 10000.0;

// reading parameter file

    if((fpr=fopen("svm_learn.par","rt"))==NULL){
       printf("Can not open parameter file : svm_learn.par\n");
       exit(1);
    }
    strcpy(pos_doc_file,"");
    strcpy(neg_doc_file,"");
    strcpy(model_file,"");
    while((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)) {
       pre_process_string(tmp_line);
       if(tmp_line[0] == '#') continue ;
       sscanf(tmp_line,"%s",str1);
       pos=0;
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       sscanf(tmp_line+pos,"%s",str2);
       if(strcmp(str1,"positive-set:")==0)strcpy(pos_doc_file,str2);
       if(strcmp(str1,"negative-set:")==0)strcpy(neg_doc_file,str2);
       if(strcmp(str1,"output-model-file:")==0)strcpy(model_file,str2);

//       if(strcmp(str1,)==0)strcpy(,str2);
      printf("%s  %s\n",str1,str2);
      if(strcmp(str1,"auto-scaling:")==0)auto_scal=atoi(str2);
      if(strcmp(str1,"visibility:")==0)visible=atoi(str2);
	  if(strcmp(str1,"kernel:")==0){
		  kernel_t=atoi(str2);
	  }
      if(strcmp(str1,"nu:")==0){nu=atof(str2);printf("nu=%16.6lf\n",nu);}
      if(strcmp(str1,"j-value:")==0){J_nu=atof(str2);printf("J_nu=%16.6lf\n",J_nu);}
      if(strcmp(str1,"p_rate:")==0){p_rate=atof(str2);printf("p_rate=%8.4lf\n",p_rate);}
      if(strcmp(str1,"n_rate:")==0){n_rate=atof(str2);printf("n_rate=%8.4lf\n",n_rate);}
      if(strcmp(str1,"gamma-value:")==0){gamma_val=atof(str2);printf("gamma_val=%16.14lf\n",gamma_val);}
      if(strcmp(str1,"stop_criteria:")==0){stop_criteria = atof(str2);printf("stop_criteria=%16.14lf\n",stop_criteria);}
	  if(strcmp(str1,"debug:")==0){
		  if(atoi(str2) == 1){
			  debug = true;
		  }else{
			  debug = false;
		  }
	  }
    }
    fclose(fpr);
//
//  reading positive data file
//

    if((fpr=fopen(pos_doc_file,"rt"))==NULL){
       printf("Can not open parameter file : %s\n",pos_doc_file);
       exit(1);
    }

    while((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)) {
       pre_process_string(tmp_line);
       if(tmp_line[0] != '#') break;
    }
    printf("checking positive data . . . . . . ");fflush(stdout);
    sscanf(tmp_line,"%d",&i);
    pos=0;
    while(tmp_line[pos]==' ') pos++;
    while(tmp_line[pos] >' ') pos++;
    sscanf(tmp_line+pos,"%s",str1);
    while(tmp_line[pos]==' ') pos++;
    while(tmp_line[pos]>' ') pos++;
    num_var=0;
    while(!(sscanf(tmp_line+pos,"%lf",&tt) == EOF)){
         while(tmp_line[pos]==' ') pos++;
	 while(tmp_line[pos]>' ') pos++;
	 num_var++;
    }

    num_p=1;
    while((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)) {
       pre_process_string(tmp_line);
       if(tmp_line[0] == '#') continue ;
       sscanf(tmp_line,"%d",&i);
       pos=0;
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       sscanf(tmp_line+pos,"%s",str1);
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       kk=0;
       while(!(sscanf(tmp_line+pos,"%lf",&tt) == EOF)){
	  while(tmp_line[pos]==' ') pos++;
	  while(tmp_line[pos]>' ') pos++;
	  kk++;
       }
       if(kk!=num_var){
	  printf("the dimension number of this line %d %s is %d not %d \n",i,str1,kk,num_var);
	  exit(1);
       }
       num_p++;
    }

    printf("done!\n");fflush(stdout);
    printf("reading into memory . . . . . . ");fflush(stdout);
    docs_p = new DOC [num_p];
    rewind(fpr);
    mm=0;
    while((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)) {
       pre_process_string(tmp_line);
       if(tmp_line[0] == '#') continue ;
       docs_p[mm].p=1;
       sscanf(tmp_line,"%d",&i);
       docs_p[mm].ind = i ;
       pos=0;
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       sscanf(tmp_line+pos,"%s",str1);
       strcpy(docs_p[mm].nam, str1);
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       kk=0;
       docs_p[mm].weight = new double [num_var];

       while(!(sscanf(tmp_line+pos,"%lf",&tt) == EOF)){
	  docs_p[mm].weight[kk] = tt ;
	  while(tmp_line[pos]==' ') pos++;
	  while(tmp_line[pos]>' ') pos++;
	  kk++;
       }
       mm++;
       if(( mm % 100)==0 ) printf("%d ",mm);fflush(stdout);
    }
    printf("%d done!\n",mm);fflush(stdout);
    fclose(fpr);

//
//  reading negative data file
//
    if((fpr=fopen(neg_doc_file,"rt"))==NULL){
       printf("Can not open parameter file : %s\n",neg_doc_file);
       exit(1);
    }
    printf("checking negative data . . . . . . ");fflush(stdout);
    num_n=0;
    while((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)) {
       pre_process_string(tmp_line);
       if(tmp_line[0] == '#') continue ;
       sscanf(tmp_line,"%d",&i);
       pos=0;
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       sscanf(tmp_line+pos,"%s",str1);
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       kk=0;
       while(!(sscanf(tmp_line+pos,"%lf",&tt) == EOF)){
	  while(tmp_line[pos]==' ') pos++;
	  while(tmp_line[pos]>' ') pos++;
	  kk++;
       }
       if(kk!=num_var){
	  printf("the dimension number of this line %d %s is %d not %d \n",i,str1,kk,num_var);
	  exit(1);
       }
       num_n++;
    }

    printf("done!\n");fflush(stdout);
    printf("reading into memory . . . . . . ");fflush(stdout);

    docs_n = new DOC [num_n];
    rewind(fpr);
    mm=0;
    while((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)) {
       pre_process_string(tmp_line);
       if(tmp_line[0] == '#') continue ;
       docs_n[mm].p=-1;
       sscanf(tmp_line,"%d",&i);
       docs_n[mm].ind = i ;
       pos=0;
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       sscanf(tmp_line+pos,"%s",str1);
       strcpy(docs_n[mm].nam, str1);
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;
       kk=0;
       docs_n[mm].weight = new double [num_var];

       while(!(sscanf(tmp_line+pos,"%lf",&tt) == EOF)){
	  docs_n[mm].weight[kk] = tt ;
	  while(tmp_line[pos]==' ') pos++;
	  while(tmp_line[pos]>' ') pos++;
	  kk++;
       }
       mm++;
       if(( mm % 100)==0 ) printf("%d ",mm);fflush(stdout);
    }
    printf(" %d done!\n",mm);fflush(stdout);
    fclose(fpr);
/*
    printf("\n%d %s %d \n",docs_p[num_p-1].ind, docs_p[num_p-1].nam, docs_p[num_p-1].p);
    for(j=0;j<num_var;j++)printf("%2.0lf ",docs_p[num_p-1].weight[j]);
    printf("\n%d %s %d \n",docs_n[num_n-1].ind, docs_n[num_n-1].nam, docs_n[num_n-1].p);
    for(j=0;j<num_var;j++)printf("%2.0lf ",docs_n[num_n-1].weight[j]);
*/
//
//   auto-scaling
//

    num_tot = num_p + num_n ;
    av = new double [num_var];
    sd = new double [num_var];
    printf("auto-scaling . . . . . . ");fflush(stdout);
  

    switch (auto_scal){
       case 0 : break ;
       case 1 : if((fpr=fopen("AverageSd.dat","rt"))==NULL){
		    printf("Can not open parameter file : average-sd.dat\n");
		    exit(1);
		}
		kk=0;
		while((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)) {
		       pre_process_string(tmp_line);
		       if(tmp_line[0] == '#') continue ;
		       sscanf(tmp_line,"%d",&i);
		       pos=0;
		       while(tmp_line[pos]==' ') pos++;
		       while(tmp_line[pos]>' ') pos++;
		       sscanf(tmp_line+pos,"%lf",&tt);
		       av[kk]=tt;
		       while(tmp_line[pos]==' ') pos++;
		       while(tmp_line[pos]>' ') pos++;
		       sscanf(tmp_line+pos,"%lf",&tt) ;
		       sd[kk]=tt;
		       kk++;
		       if(kk>num_var){printf("average-sd.dat file not correct!\n");exit(0);}
		}
		fclose(fpr);


		for( i=0; i<num_var; i++){
		    for(j=0;j<num_p;j++)if(sd[i]!=0.0)docs_p[j].weight[i]=(docs_p[j].weight[i] -av[i] )/sd[i];
		    for(j=0;j<num_n;j++)if(sd[i]!=0.0)docs_n[j].weight[i]=(docs_n[j].weight[i] -av[i] )/sd[i];
		}
		
		break;
       case 2 : tt = (double) num_tot;
		xx = tt - 1.0 ;
//		printf("num_var =%d  \n",num_var);fflush(stdout);
		for(i=0;i<num_var;i++){
		   av[i]=0.0;
		   for(j=0;j<num_p;j++)av[i]=av[i]+docs_p[j].weight[i];
		   for(j=0;j<num_n;j++)av[i]=av[i]+docs_n[j].weight[i];
		   av[i]=av[i]/tt;
		   sd[i]=0;
		   for(j=0;j<num_p;j++)sd[i]=sd[i]+(docs_p[j].weight[i]-av[i])*(docs_p[j].weight[i]-av[i]);
		   for(j=0;j<num_n;j++)sd[i]=sd[i]+(docs_n[j].weight[i]-av[i])*(docs_n[j].weight[i]-av[i]);
		   sd[i]=sd[i]/xx;
		   sd[i]=sqrt(sd[i]);
		}
		
		for( i=0; i<num_var; i++){
		    for(j=0;j<num_p;j++)if(sd[i]!=0.0)docs_p[j].weight[i]=(docs_p[j].weight[i] -av[i] )/sd[i];
		    for(j=0;j<num_n;j++)if(sd[i]!=0.0)docs_n[j].weight[i]=(docs_n[j].weight[i] -av[i] )/sd[i];
		}
		if( (fpw=fopen("average_sd.dat","wt")) == NULL){
		   printf("open average_sd.dat err!\n"); 
		   fflush(stdout);
		}
		else {
		   fprintf(fpw," #index   average        sd\n"); fflush(stdout);
		   for( i=0; i<num_var; i++){
		       k=i+1;
		       fprintf(fpw,"%6d %12.8lf  %12.8lf\n",k,av[i],sd[i]);
		
		   }
		   fclose(fpw);
		}

		break;
    }
    printf("done!\n");fflush(stdout);
/*
    printf("\n%d %s %d \n",docs_p[num_p-1].ind, docs_p[num_p-1].nam, docs_p[num_p-1].p);
    for(j=0;j<num_var;j++)printf("%6.4lf ",docs_p[num_p-1].weight[j]);
    printf("\n%d %s %d \n",docs_n[num_n-1].ind, docs_n[num_n-1].nam, docs_n[num_n-1].p);
    for(j=0;j<num_var;j++)printf("%6.4lf ",docs_n[num_n-1].weight[j]);
*/
    delete[] av;
    delete[] sd;

//
//   randommizing the positive and negative data sets respectively
//

//    printf("\nNative  before shuffling!\n");
//    for (i=0;i<num_p;i++) printf("%d ",docs_p[i].ind); printf("\n"); fflush(stdout);
  
//    printf("\nDecoy  before shuffling!\n");
//   for (i=0;i<num_n;i++) printf("%d ",docs_n[i].ind); printf("\n"); fflush(stdout);
    arr_t = new int [num_tot];
    random_1_nn(num_p,arr_t);
    
//    printf("\nfisrt random:"); for (i=0;i<num_p;i++) printf("%d ",arr_t[i]); printf("\n"); fflush(stdout);
    
    if (num_p>num_n) mm=num_p;
    else mm = num_n;
    docs_t = new DOC [mm];
    for(i=0;i<mm;i++)docs_t[i].weight = new double[num_var];
    for(i=0;i<num_p;i++){
	docs_t[i].ind=docs_p[arr_t[i]-1].ind;
	strcpy(docs_t[i].nam,docs_p[arr_t[i]-1].nam);
	docs_t[i].p=docs_p[arr_t[i]-1].p;
	for(j=0;j<num_var;j++)docs_t[i].weight[j]=docs_p[arr_t[i]-1].weight[j];
//       printf("\n%d  %s  %d\n",docs_t[i].ind,docs_t[i].nam,docs_t[i].p);
//       for(j=0;j<num_var;j++)printf("%lf ",docs_t[i].weight[j]);
//       printf("\n\n%d  %s  %d\n",docs_p[arr_t[i]-1].ind,docs_p[arr_t[i]-1].nam,docs_p[arr_t[i]-1].p);
//       for(j=0;j<num_var;j++)printf("%lf ", docs_p[arr_t[i]-1].weight[j]);
    }
    for(i=0;i<num_p;i++){
       docs_p[i].ind=docs_t[i].ind;
       strcpy(docs_p[i].nam,docs_t[i].nam);
       docs_p[i].p=docs_t[i].p;
       for(j=0;j<num_var;j++)docs_p[i].weight[j]=docs_t[i].weight[j];
    }
    random_1_nn(num_n,arr_t);
    
//    printf("\nsecond random:"); for (i=0;i<num_n;i++) printf("%d ",arr_t[i]); printf("\n"); fflush(stdout);
    
    for(i=0;i<num_n;i++){
       docs_t[i].ind=docs_n[arr_t[i]-1].ind;
       strcpy(docs_t[i].nam,docs_n[arr_t[i]-1].nam);
       docs_t[i].p=docs_n[arr_t[i]-1].p;
       for(j=0;j<num_var;j++)docs_t[i].weight[j]=docs_n[arr_t[i]-1].weight[j];
    }
    for(i=0;i<num_n;i++){
       docs_n[i].ind=docs_t[i].ind;
       docs_n[i].p=docs_t[i].p;
       strcpy(docs_n[i].nam,docs_t[i].nam);
       for(j=0;j<num_var;j++)docs_n[i].weight[j]=docs_t[i].weight[j];
    }

//    printf("\nNative  after shuffling!\n");
//    for (i=0;i<num_p;i++) printf("%d ",docs_p[i].ind); printf("\n"); fflush(stdout);
//    printf("\nDecoy  after shuffling!\n");
    
    delete[] arr_t;
    for(i=0;i<mm;i++)delete[] docs_t[i].weight;
    delete[] docs_t;

    svm_learn(docs_p,docs_n);

   for(i=0;i<num_p;i++) delete[] docs_p[i].weight;
   delete[] docs_p;
   for(i=0;i<num_n;i++) delete[] docs_n[i].weight;
  delete[] docs_n;

    return 0;
}

void svm_learn(DOC *d_p, DOC *d_n){
    int i,j,k,m,ii,jj,kk,i1,i2;
	int num_pre_sele_SVs_p, *pre_sele_SVs_p, num_pre_sele_SVs_n, *pre_sele_SVs_n, *ii_tmp;
    double *dataset, *SVs_set, *KM, *alpha, *b_gamma, tt,tr_err;
//	double *dbar;
//
//   building dataset
//
//
    dataset = new double [num_tot*num_var];
    for(i=0;i<num_p;i++)
       for(j=0;j<num_var;j++)dataset[i*num_var+j]=d_p[i].weight[j];
    for(i=0;i<num_n;i++)
       for(j=0;j<num_var;j++)dataset[(i+num_p)*num_var+j]=d_n[i].weight[j];
//    printf("Coming here 1 !\n"); fflush(stdout);
//
// building Support Vectors set SVs_set
//
    num_SVs_p = (int) floor(p_rate*num_p);
    num_SVs_n = (int) floor(n_rate*num_n);

	pre_sele_SVs_p = new int [num_tot];
	pre_sele_SVs_n = new int [num_tot];
	ii_tmp = new int [num_tot];

	num_pre_sele_SVs_p = 0 ;
    if ( (fpr=fopen("pre_select_SVs_p.file","rt")) != NULL ) {
		fscanf(fpr, "%d", &kk);
		while ( !feof(fpr) ) {
			pre_sele_SVs_p[num_pre_sele_SVs_p] = kk ;
			num_pre_sele_SVs_p++;
			fscanf(fpr, "%d", &kk);
		}
   }
	if ( num_pre_sele_SVs_p == 0 ) {
		printf("  No pre-selected positive SVs !\n");
	} else {
		if ( num_pre_sele_SVs_p > num_SVs_p ) {
			printf(" Warning !!! --- the number of pre-selected positive SVs() | > | that  setup by .par file !\n");
			printf("  and tot_SVs_p was re-setup !\n");
			num_SVs_p =  num_pre_sele_SVs_p ;
			printf("  total pre-selected positive SVs  is %d!\n", num_pre_sele_SVs_p);
			fflush(stdout);
		} else {
			printf("  total pre-selected positive SVs  is %d!\n", num_pre_sele_SVs_p);
			fflush(stdout);
		}
	}


	num_pre_sele_SVs_n = 0 ;
    if ( (fpr=fopen("pre_select_SVs_n.file","rt")) != NULL ) {
		fscanf(fpr, "%d", &kk);
		while ( !feof(fpr) ) {
			pre_sele_SVs_n[num_pre_sele_SVs_n] = kk ;
			num_pre_sele_SVs_n++;
			fscanf(fpr, "%d", &kk);
		}
   }
	if ( num_pre_sele_SVs_n == 0 ) {
		printf("  No pre-selected positive SVs !\n");
	} else {
		if ( num_pre_sele_SVs_n > num_SVs_n ) {
			printf(" Warning !!! --- the number of pre-selected negative SVs() | > | that  setup by .par file !\n");
			printf("  and tot_SVs_n was re-setup !\n");
			num_SVs_n = num_pre_sele_SVs_n;
			printf("  total pre-selected negative SVs  is %d!\n", num_pre_sele_SVs_n);
			fflush(stdout);
		} else {
			printf("  total pre-selected positive SVs  is %d!\n", num_pre_sele_SVs_n);
			fflush(stdout);
		}
	}


    num_SVs_tot = num_SVs_p + num_SVs_n ;
    SVs_set = new double [num_SVs_tot*num_var];
    
    SVs_p = new int [num_SVs_p];
    random_1_nn(num_p,ii_tmp);
    

//    printf("\nthird random:"); for (i=0;i<num_p;i++) printf("%d ",ii_tmp[i]); printf("\n"); fflush(stdout);
    
	kk=0;
	jj=0;
	i1=0;
	i2=0;
	ii = num_SVs_p - num_pre_sele_SVs_p ;
	while ( kk < num_p ) {
		k = ii_tmp[kk] - 1 ;
		j = d_p[k].ind ;
		m = -9 ;
		for ( i=0; i<num_pre_sele_SVs_p; i++) {
			if ( j == pre_sele_SVs_p[i]) {
				m = 9 ;
				break ;
			}
		}
		if ( (m == -9 ) && (i1<ii ) ) {
			SVs_p[jj] = k ;
			i1++ ;
			jj++;
		}
		if ( (m == 9 ) && ( i2<num_pre_sele_SVs_p ) ) {
			SVs_p[jj] = k ;
			i2++ ;
			jj++;
		}
		if ( (i1==ii) && (i2==num_pre_sele_SVs_p) ) break ;
		kk++;
	}


    for(i=0;i<num_SVs_p;i++)
       for(j=0;j<num_var;j++)SVs_set[i*num_var+j]=d_p[SVs_p[i]].weight[j];

   printf("No. of native SVs - %d ! ",num_SVs_p); 

// for (i=0;i<num_SVs_p;i++) printf("%d ",d_p[SVs_p[i]].ind); 
printf("\n"); fflush(stdout);

SVs_n = new int [num_SVs_n];
    random_1_nn(num_n,ii_tmp);
 
    //   printf("\nforth random:"); for (i=0;i<num_n;i++) printf("%d ",ii_tmp[i]); printf("\n"); fflush(stdout);

    kk=0;
	jj=0;
	i1=0;
	i2=0;
	ii = num_SVs_n - num_pre_sele_SVs_n ;
	while ( kk < num_n ) {
		k = ii_tmp[kk] - 1 ;
		j = d_n[k].ind ;
		m = -9 ;
		for ( i=0; i<num_pre_sele_SVs_n; i++) {
			if ( j == pre_sele_SVs_n[i]) {
				m = 9 ;
				break ;
			}
		}
		if ( (m == -9 ) && (i1<ii ) ) {
			SVs_n[jj] = k ;
			i1++ ;
			jj++;
		}
		if ( (m == 9 ) && ( i2<num_pre_sele_SVs_n ) ) {
			SVs_n[jj] = k ;
			i2++ ;
			jj++;
		}
		if ( (i1==ii) && (i2==num_pre_sele_SVs_n) ) break ;
		kk++;
	}



    for(i=0;i<num_SVs_n;i++)
       for(j=0;j<num_var;j++)SVs_set[(i+num_SVs_p)*num_var+j]=d_n[SVs_n[i]].weight[j];

    printf("No. of decoy SVs - %d ! ",num_SVs_n); 
//    for (i=0;i<num_SVs_n;i++) printf("%d ",d_n[SVs_n[i]].ind);
    printf("\n"); fflush(stdout);
   delete[] pre_sele_SVs_p ;
   delete[] pre_sele_SVs_n ;
   delete[] ii_tmp;

//    printf("Coming here 2 ! %d %d \n",num_SVs_p, num_SVs_n); fflush(stdout);
//    exit(0);
//
//  building dbar,KM matrices
//
    printf("gereating kernel matrix(KM) . . . . . . "); fflush(stdout);
//    dbar = new double[num_tot*num_tot];
//    matrix_gen(num_tot,num_tot,0.0,dbar);
//    for(i=0;i<num_p;i++)dbar[i*num_tot+i] = 1.0;
//    for(i=0;i<num_n;i++)dbar[(i+num_p)*num_tot+i+num_p] = -1.0;

// reading debug-test data
/*
	num_var=14;
	num_SVs_tot=14;
	num_tot=274;
	num_p=148; num_n=126;

	fpr=fopen("ATrain.dat","rt");
	for(i=0;i<274;i++)
		for(j=0;j<14;j++){
			fscanf(fpr,"%lf",&tt);
			dataset[i*14+j]=tt;
		}
	fclose(fpr);


	fpr=fopen("RAset.dat","rt");
	for(i=0;i<14;i++)
		for(j=0;j<14;j++){fscanf(fpr,"%lf",&tt);SVs_set[i*14+j]=tt;}
	fclose(fpr);

*/
// end reading debug-test data
    KM = new double [num_tot*num_SVs_tot];
    for(i=0;i<num_tot;i++){
		ii = i*num_var;
		for(j=0;j<num_SVs_tot;j++){
			jj=j*num_var;
			tt=0.0;
			for(k=0;k<num_var;k++){
				tr_err=dataset[ii+k]-SVs_set[jj+k];
				tt=tt+tr_err*tr_err;
			}
			KM[i*num_SVs_tot+j]=exp(-2.0*gamma_val*tt);
//			printf("%8.4lf",KM[i*num_SVs_tot+j]);
		}
//		printf("\n");
	}
    printf("done !\n"); fflush(stdout);
//
//

    alpha = new double[1+num_tot];
    b_gamma = new double [2];

    n_svm(KM,alpha,b_gamma,d_p,d_n);
//
//	Output model file
//
	fpw=fopen(model_file,"wt");
	fprintf(fpw,"SVM-reduced learning for %s + %s\n",pos_doc_file,neg_doc_file);
	fprintf(fpw," %d  # kernel type\n",kernel_t);
	fprintf(fpw," %16.12lf # kernel parameter -g\n",gamma_val);
	fprintf(fpw," %d  # feature number\n",num_var);
	fprintf(fpw," %d  # number of training documents\n",num_tot);
	fprintf(fpw," %d  # number of positive support vectors\n",num_SVs_p);
	fprintf(fpw," %d  # number of negative support vectors\n",num_SVs_n);
        fprintf(fpw," %d  # auto-scaling\n",auto_scal);
	fprintf(fpw," %16.6lf  # nu value\n",nu);
	fprintf(fpw," %16.6lf  # J  value\n",J_nu);
	fprintf(fpw," %20.16lf #b-gamma value\n",b_gamma[1]);
    for(i=0;i<num_SVs_p;i++){
		k=SVs_p[i];
		for(j=0;j<num_var;j++)fprintf(fpw,"%10.8lf ",d_p[k].weight[j]);
		fprintf(fpw,"\n %lf  # %d %s \n",alpha[i],d_p[k].ind,d_p[k].nam);
	}
    for(i=0;i<num_SVs_n;i++){
		k=SVs_n[i];
		for(j=0;j<num_var;j++)fprintf(fpw,"%10.8lf ",d_n[k].weight[j]);
		fprintf(fpw,"\n %lf  # %d %s \n",alpha[i+num_SVs_p],d_n[k].ind,d_n[k].nam);
	}
	fclose(fpw);
//
//
   delete[] b_gamma;
   delete[] KM;
   delete[] SVs_set;
   delete[] dataset;
//   delete[] dbar;
   delete[] SVs_p;
   delete[] SVs_n;
   delete[] alpha;


    return;
}
void n_svm(double *A,  double *w, double *gamma, DOC *d_p, DOC *d_n){
   int i,j,k,m,n,*im,in,jn,nn,iter;
   double alpha, *H, *Q, xx,current_xx;
//   double **a, **v, *star, *hu, *u;
   double  *star, *hu, *u;
   int ldu,ldvt,lwork,lda,ldb,info,*ipiv,emm,enn,rsh=1;
   double *s,*ut,*vt,*work, *HH;
   char ch1='N'; //str[99],
//
//

   m=num_tot;
   lda=NM_MAX;
   ldb=lda;
   n=num_SVs_tot;
   im = new int [m+4];
   for(i=0;i<(m+4);i++)im[i]=i*m;
//
//  create H & Q matrices
//

   H = new double [(m+1)*(m+1)];
   Q = new double [(m+1)*(m+1)];
   nn=n+1;
   for(i=0;i<num_p;i++){
      k=i*nn;
	  in=i*n;
      for(j=0;j<n;j++)H[k+j]=A[in+j];
      H[k+n]=-1.0;
   }
   for(i=0;i<num_n;i++){
      k=(i+num_p)*nn;
	  in=(i+num_p)*n;
      for(j=0;j<n;j++)H[k+j]=-A[in+j];
      H[k+n]=1.0;
   }
   for(i=0;i<m;i++){
      in=i*nn;
      for(j=0;j<m;j++){
		  if(i==j)  {  Q[im[i]+j] = 1.0 / nu;  if ( i<num_p ) Q[im[i]+j] = Q[im[i]+j] / J_nu; }
		  else Q[im[i]+j]=0.0;
		  jn=j*nn;
		  for(k=0;k<nn;k++) Q[im[i]+j]=Q[im[i]+j]+H[in+k]*H[jn+k];
//		  printf("%8.4lf",Q[im[i]+j]);
      }
//	  printf("\n");
   }

lwork = 8*lda;

    HH = (double *)malloc((lda*m)*sizeof(double));
    hu = (double *)malloc(ldb*sizeof(double));
    ipiv = (int *)malloc(lda*sizeof(int));
    s = (double *)malloc((lda)*sizeof(double));
    ut = (double *)malloc(sizeof(double));
    vt = (double *)malloc(sizeof(double));
    work = (double *)malloc((lwork)*sizeof(double));
		 

//    a=H', a=UWV; w is svd(a); after call svdcmp a was changed to U;
// output the H' to file and run SVD program by system command
// then then reading the norm(H',2) back from SVD runbig results
emm=nn; enn=m;
/*
   if ( (fpw=fopen("Matrix_H__T.dat","wt")) != NULL ) {
        fprintf(fpw, "%d  , %d\n", nn,m);
*/


        for(i=0;i<nn;i++)
	   for(j=0;j<m;j++) {
//    fprintf(fpw,"%26.16lf\n",H[j*nn+i]);
	      HH[i+j*lda]=H[j*nn+i];
	   }


/*
fclose(fpw);
   }
*/
if(debug){
	#ifdef WIN32
	printf("Printing time before svd the %d * %d matrix: ",nn,m); fflush(stdout); system("date /T & time /T");
	#else
	printf("Printing time before svd the %d * %d matrix: ",nn,m); fflush(stdout); system("date");
	#endif
}

delete[] H;
ldu = 1;
ldvt = 1;

#ifdef WIN32
	DGESVD(&ch1,&ch1, &emm, &enn, HH, &lda, s, ut, &ldu, vt, &ldvt, work, &lwork,&info);
#else
	dgesvd_(&ch1,&ch1, &emm, &enn, HH, &lda, s, ut, &ldu, vt, &ldvt, work, &lwork,&info);
#endif


alpha = s[0] ;
if (debug){
	#ifdef WIN32
	printf("Norm(H^^T)=%20.16lf\nPrinting time after svd : ",alpha); fflush(stdout); system("date /T & time /T");
	#else
	printf("Norm(H^^T)=%20.16lf\nPrinting time after svd : ",alpha); fflush(stdout); system("date");
	#endif
}
/*
    fpw=fopen("svd_sss","wt");
    for(i=0;i<nn;i++)fprintf(fpw,"%26.16lf\n",s[i]);
    fclose(fpw);
*/
    free(s);
    free(ut);
    free(vt);
    free(work);
			     
		star = new double[1+m];
		u = new double[1+m];
//	hu = new double[1+m];
/*
system("SVD_lapack_C  Matrix_H__T.dat  SVD_of_H__T.dat");
	if ( (fpr=fopen("SVD_of_H__T.dat","rt")) != NULL ) {
	    fscanf(fpr,"%lf",&alpha);
	    fclose(fpr);
	}
*/

		alpha = alpha*alpha*1.1+1.1/nu;

//   in matlab:   hu=-max(((Q*u-e)-alpha*u),0)+Q*u-e;
//      at this step, hu = [-1](1:m) column vector;

		for(i=0;i<m;i++){
//	a[i][1] =-1.0;
			u[i] = 0.0;
//	hu[i]=a[i][1];
			hu[i] = - 1.0;
		}
		
		xx = vector_norm(m,hu);

//		printf("m= %d | hu[0]=%lf | norm(hu)=%lf\n",m,hu[0],xx);

		iter=1;
		current_xx=1.0e99;
		while( (xx>stop_criteria) && ( fabs(current_xx-xx) >= stop_criteria ) ){
		        current_xx = xx;
			for(i=0;i<m;i++){
				star[i]=(Q[im[i]+i]-alpha)*u[i]-1;
				for(j=0;j<i;j++)star[i]=star[i]+Q[im[i]+j]*u[j];
				for(j=i+1;j<m;j++)star[i]=star[i]+Q[im[i]+j]*u[j];
				if(star[i]>TINY) star[i]=1.0;
				else star[i]=0.0;
			}
/*
        if(iter<10)sprintf(str,"Matrix_A_and_B_0%d.dat",iter);
	        else sprintf(str,"Matrix_A_and_B_%d.dat",iter);
fpw=fopen(str,"wt") ;  fprintf(fpw, "%d\n", m); 
*/			
			for(i=0;i<m;i++){
//	for(j=1;j<=m;j++)a[i][j]=Q[im[i-1]+j-1]*(1-star[i]);
				for(j=0;j<m;j++)w[j]=Q[im[i]+j]*(1-star[i]);
//	a[i][i]=a[i][i]+alpha*star[i];
				w[i]=w[i]+alpha*star[i];
				for(j=0;j<m;j++){ 
//      fprintf(fpw,"%26.16lf\n",w[j]); 
				      HH[i+j*lda] = w[j] ;
				      }
			}
/*
    for ( i=0;i<m;i++ ){
            for(j=0;j<m;j++){
                fprintf(fpw,"%20.14lf\n",HH[i+j*lda]);
        }
    }
       for(i=0;i<m;i++){
	  fprintf(fpw,"%20.14lf\n",hu[i]);
	  }
       fclose(fpw);
*/    
if (debug){
	#ifdef WIN32
	printf("\nPrinting time before solving the %d-D linear equation: ",m); fflush(stdout); system("date /T & time /T");
	#else
	printf("\nPrinting time before solving the %d-D linear equation: ",m); fflush(stdout); system("date");
	#endif
}
// system("SOL_lin_eq_lapack_C  Matrix_A_and_B.dat  X_of_AX_eq_B.dat");

enn=m;
rsh=1;

#ifdef WIN32
	DGESV( &enn, &rsh, HH, &lda, ipiv, hu, &ldb, &info);
#else
	dgesv_( &enn, &rsh, HH, &lda, ipiv, hu, &ldb, &info);
#endif



/*
fpw=fopen("xxxx","wt");
    for(i=0;i<m;i++)fprintf(fpw,"%26.16lf\n",hu[i]);
   fclose(fpw);
if ( (fpr=fopen("X_of_AX_eq_B.dat","rt")) != NULL ) {
      for(i=0;i<m;i++)
          fscanf(fpr,"%lf",&hu[i]);
      fclose(fpr);
   }
*/
if (debug){
	#ifdef WIN32
	printf("Printing time after solving  equation: "); fflush(stdout); system("date /T & time /T");
	#else
	printf("Printing time after solving  equation: "); fflush(stdout); system("date");
	#endif
}

			for(i=0;i<m;i++)
				u[i]=u[i]-hu[i];
			for(i=0;i<m;i++){
				hu[i]=-1.0;
				for(j=0;j<m;j++)
					hu[i]=hu[i]+u[j]*Q[im[i]+j];
				xx=hu[i]-u[i]*alpha;
				if(xx<TINY)xx=0.0;
				hu[i]=hu[i]-xx;
//	a[i][1]=hu[i];
//	printf("%10.4lf\n",hu[i]);
			}
			
			xx = vector_norm(m,hu);
			printf("iteration %d : xx = %1.2le\n",iter,xx); fflush(stdout);
			iter++;
		}

		printf(" iteration done !\n"); fflush(stdout);
		gamma[1]=0.0;
		for(i=0;i<num_p;i++) gamma[1]=gamma[1]+u[i];
		for(i=num_p;i<m;i++){
			u[i]=-u[i];
			gamma[1]=gamma[1]+u[i];
		}

		gamma[1]=-gamma[1];

		for(i=0;i<n;i++){
			w[i]=0.0;
			for(j=0;j<m;j++)
				w[i]=w[i]+A[j*n+i]*u[j];
//			printf("%lf\n",w[i]);
		}
// Classification for the traing set


		k=0;
		for(i=0;i<num_p;i++){
			xx=-gamma[1];;
			for(j=0;j<num_SVs_tot;j++)xx=xx+w[j]*A[i*num_SVs_tot+j];
			if(xx>0.0)k++;
		}
		printf("Total %d  of  %d  positive were classified correctly\n",k,num_p);fflush(stdout);

/*
		nn=0;
		fpw=fopen("tmp.de.pre","wt");
		for(i=0;i<num_n;i++){
			xx=-gamma[1];;
			for(j=0;j<num_SVs_tot;j++)xx=xx+w[j]*A[(i+num_p)*num_SVs_tot+j];
			strcpy(str,d_n[i].nam);
			if(str[6]=='.') str[6]='\0';
			else str[4]='\0';
			fprintf(fpw,"%d %s %lf\n",d_n[i].ind,str,xx);
			if(xx<0.0)nn++;
		}
		printf("Total %d  of  %d  negative were classified correctly\n",nn,num_n);fflush(stdout);
		fclose(fpw);
*/
		nn=0;
		for(i=0;i<num_n;i++){
			xx=-gamma[1];;
			for(j=0;j<num_SVs_tot;j++)xx=xx+w[j]*A[(i+num_p)*num_SVs_tot+j];
			if(xx<0.0)nn++;
		}
		printf("Total %d  of  %d  negative were classified correctly\n",nn,num_n);fflush(stdout);

		nn=nn+k;
		xx = (double) nn / (double) num_tot;
		xx=xx*100.0;
		printf("Total classification rate %d / %d (%6.2lf %% )\n",nn,num_tot,xx); fflush(stdout);

		delete[] star;
		free(hu);
		delete[] u;
		delete[] Q;
		free(HH);
//		free(bb);
		free(ipiv);
			     
}

// discard the beginning and ending space, tab, linefeed and
// other <14 char
void pre_process_string(char *str){
   int i,k,j,m;
   char *temp;

   k = strlen(str);
   m=k+9;
   temp = new char[m];
   i=str[k-1];
   while((i==32)||(i<14)){
      k--;
      if(k==0)break;
      i=str[k-1];
   }
   str[k]='\0';
   j=0;
   i=str[j];
   while((i==32)||(i<14)){
      j++;
      if(j==k)break;
      i=str[j];
   }
   if(j>0){
      m=k-j;
      for(i=0; i<m; i++)temp[i]=str[j+i];
      temp[i]='\0';
      strcpy(str,temp);
   }
   delete[] temp;
};

// generating random series between 1 and nn;
void random_1_nn(int nn, int *arr){

   int j,k,kk,mm,nnn;

   kk = time(NULL) + rand() ;
   srand( (unsigned)kk );
   nnn=nn+1;
   mm=0;
   while(mm<nn){
      kk=rand();
      while(!((kk>0)&&(kk<nnn))) {
         kk=rand();
         if(kk>nn)kk=kk%nnn;
      }
      k=0;
      for(j=0;j<mm;j++)if(kk==arr[j]){k=1;break;}
      if(k==0){
         arr[mm]=kk;
         mm++;
      }
   }
}

//norm of vector is the sqrt(sum(x(i)^2))
double vector_norm(int nn, double *xx){
	int i;
	double s=0.0;
	for(i=0;i<nn;i++) s = s + xx[i]*xx[i];
	return sqrt(s);
}


