/************************************************************************/
/*                                                                      */
/*    rdsvm_cl.cpp                                                      */
/*                                                                      */
/*    Using reduced SVM function to predict native or decoy             */
/*                                                                      */
/*    Author: Yun Xu                                                    */
/*    Date: Apr 25, 2009                                                */
/*                                                                      */
/*    Copyright (c) 2000  University of Illinois - All rights reserved  */
/*                                                                      */
/*    This software is available for non-commercial use only. It must   */
/*    not be modified and distributed without prior permission of the   */
/*    author. The author is not responsible for implications from the   */
/*    use of this software.                                             */
/*                                                                      */
/* **********************************************************************/

# include <stdio.h>
# include <stdlib.h>
# include <ctype.h>
# include <math.h>
# include <string.h>
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)


#define MAX_LINE_LEN 19999

char tmp_line[MAX_LINE_LEN],str1[MAX_LINE_LEN];
int num_var, num_SVs_p, num_SVs_n, num_SVs_tot, *II, auto_scal;
double  gamma_val, kernel_t, *SVs_set, *alpha , b_gamma, *KM;
FILE *fpr, *fpw, *fp1;

typedef struct sampe_doc {
  long    ind;      //index number
  char    nam[66];  // sample name
  int     p;        // 1 for positive sample; -1 for negative sample;
  double *weight;
} DOC;


void pre_process_string(char *str);
double svm_predict(DOC samp);

char pdb_nam_ar[9999][8],temp[99];
int  residu_num_ar[9999];
int  TOTAL;


int main (int argc, char* argv[]){

	int i, kk, mm, pos; 
    double tt;
    double *av, *sd;
    DOC samp;


// reading model file

	if ( argc < 5 ) { 
		printf("rdsvm_cl  sample-file  model-file AverageSd-file output-file\n");
		exit(0);
	}
			      
    if((fpr=fopen(argv[2],"rt"))==NULL){
       printf("Can not open model file : %s\n",argv[2]);
	   fflush(stdout);
       exit(1);
    }
    printf("reading modle file . . .  \n");


	fgets(tmp_line,MAX_LINE_LEN,fpr);
//	fprintf(fpw,"SVM-reduced learning for %s + %s\n",pos_doc_file,neg_doc_file);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
	sscanf(tmp_line,"%d",&kernel_t);
//	fprintf(fpw," %d  # kernel type\n",kernel_t);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
	sscanf(tmp_line,"%lf",&gamma_val);
//	fprintf(fpw," %lf # kernel parameter -g\n",gamma_val);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
	sscanf(tmp_line,"%d", &num_var);
//	fprintf(fpw," %d  # feature number\n",num_var);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
//	sscanf(tmp_line,"%d",num_tot);
//	fprintf(fpw," %d  # number of training documents\n",num_tot);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
	sscanf(tmp_line,"%d",&num_SVs_p);
//	fprintf(fpw," %d  # number of positive support vectors\n",num_SVs_p);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
	sscanf(tmp_line,"%d",&num_SVs_n);
//	fprintf(fpw," %d  # number of negative support vectors\n",num_SVs_n);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
	sscanf(tmp_line,"%d",&auto_scal);
//	printf(" %d  # auto-scaling \n",auto_scal);
	

	fgets(tmp_line,MAX_LINE_LEN,fpr);
//	sscanf(tmp_line,"%lf",&nu);
//	fprintf(fpw," %lf  # nu value\n",nu);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
//	sscanf(tmp_line,"%lf",&J_nu);
//	fprintf(fpw," %lf  # J_\n",J_nu);

	fgets(tmp_line,MAX_LINE_LEN,fpr);
	sscanf(tmp_line,"%lf", &b_gamma);
//	printf(" %20.26lf #b-gamma value\n",b_gamma);
//exit(0);
//
    num_SVs_tot = num_SVs_p + num_SVs_n ;
    SVs_set = new double [num_SVs_tot*num_var+4];
    alpha = new double [num_SVs_tot+4];

    II = new int [num_SVs_tot+4];
    for ( i=0; i<=num_SVs_tot; i++) II[i] = num_var*i;
	

    for(i=0;i<num_SVs_tot;i++){
		if((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)){
			pre_process_string(tmp_line);
			pos=0;
			kk=0;
			while(!(sscanf(tmp_line+pos,"%lf",&tt) == EOF)){
				SVs_set[i*num_var+kk]=tt;
				kk++;
				while(tmp_line[pos]==' ') pos++;
				while(tmp_line[pos]>' ') pos++;
			}
			if(kk!=num_var){
				printf("the dimension number of this line %d %s is %d not %d \n",i,tmp_line,kk,num_var);
				exit(1);
			}
 
		}else{
			printf("model file error! \n");
			fflush(stdout);
			exit(0);
		}

		if((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)){
			pre_process_string(tmp_line);
			sscanf(tmp_line,"%lf",&tt);
			alpha[i]=tt;
		}else{
			printf("model file error! \n");
			fflush(stdout);
			exit(0);
		}
	}
	fclose(fpr);


// reading average and stand deviation file
    av = new double [num_var+1];
    sd = new double [num_var+1];

if ( auto_scal >= 1 ) {
       if((fpr=fopen(argv[3],"rt"))==NULL){
		    printf("Can not open average and standard deviation file : average_sd.dat\n");
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
		       if ( kk>num_var){
			   printf("the dimension number of this line %d %s is %d not %d \n",i,tmp_line,kk,num_var);
			   exit(0);
		       }
		}
		fclose(fpr);
}

//  reading read sample file and prediction
//
    if((fpr=fopen(argv[1],"rt"))==NULL){
       printf("Can not open sample file : %s\n",argv[1]);
       exit(1);
    }

	fpw=fopen(argv[4],"wt");
	samp.weight = new double [num_var+4];
        KM = new double [num_SVs_tot+4];
	mm=0;
     
    while((!feof(fpr)) && fgets(tmp_line,MAX_LINE_LEN,fpr)) {
       pre_process_string(tmp_line);
       if(tmp_line[0] == '#') continue;
       pos=0;
       sscanf(tmp_line+pos,"%s",str1);
       strcpy(samp.nam, str1);
       while(tmp_line[pos]==' ') pos++;
       while(tmp_line[pos]>' ') pos++;

       strcpy(temp,str1); 
       if(temp[6]=='.') temp[6]='\0'; 
       else temp[4]='\0'; 
       kk=0;
       
       while(!(sscanf(tmp_line+pos,"%lf",&tt) == EOF)){
		   samp.weight[kk] = tt ;
		   while(tmp_line[pos]==' ') pos++;
		   while(tmp_line[pos]>' ') pos++;
		   kk++;
       }
       
		if( kk > num_var ) {
			printf("the dimension number of this line  %s is %d || not %d\n",tmp_line,kk,num_var);
			exit(0);
		}

		mm++;
		if(( mm % 100)==0 ) printf("%d ",mm);fflush(stdout);
       
		if ( auto_scal >= 1 ) {
			   for ( i=0;i<num_var;i++){
					if(sd[i]!=0.0) samp.weight[i]=(samp.weight[i]-av[i])/sd[i];
			   }
		}      

       tt = svm_predict(samp);
       
       fprintf(fpw,"%s\t%3.4lf\n", samp.nam,tt);
       
    }
    printf(" %d done!\n",mm);fflush(stdout);
    fclose(fpr);

    delete[] av;
    delete[] sd;
    delete[] KM;
    delete[] SVs_set;
    delete[] II;
    delete[] samp.weight;
    delete[] alpha;
    return 0;
}

double svm_predict(DOC samp){
    int j,k,jj;
    double tt,xx;

	for(j=0;j<num_SVs_tot;j++){
		jj=II[j];
		tt=0.0;
		for(k=0;k<num_var;k++){
			xx=samp.weight[k]-SVs_set[jj+k];
			tt=tt+xx*xx;
			
		}
		KM[j]=exp(-2.0*gamma_val*tt);
//		printf("%16.12lf %16.12lf  %16.12lf\n", tt, gamma_val, KM[j]);
	}

	xx=-b_gamma;
	for(j=0;j<num_SVs_tot;j++)xx=xx+alpha[j]*KM[j];
	return xx;
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

