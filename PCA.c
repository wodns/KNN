/*
PCA is implemented as a eigen value - eigen vector problem.
Dealing with large number of data, usually it contatins neally normal distributed values.
For that case, PCA eigen value - eigen vector program reduces its calculation amount along its running steps continually.
In general, PCA algorithm needs n^3 order caculations and n order memory.
However, for most cases n^3 calculations naturally reduce to nearlly n^2.
So, if you want to predict the PCA caculational time range, you must deal with n^2 term.
*/

/*input
11.0 3.4 3.5 3.6
11.0 2.2 2.3 2.4
11.0 3.1 4.3 3.8
11.0 5.2 4.3 2.9
11.0 8.3 3.4 5.2
11.0 2.1 5.4 2.2
11.0 9.0 4.1 2.1
11.0 0.2 0.1 0.8
11.0 0.8 3.2 0.8
11.0 0.8 2.3 4.2
11.0 0.9 3.2 4.3
11.0 0.7 4.2 0.8*/

/*output
Analysis of correlations chosen.

Means of column vectors:11.0 4.2 3.4 2.9
Standard deviations of columns:1.0 2.9 1.5 1.2

A
Eigenvalues:
1.90539
0.48207
0.61254
1.00000

(Eigenvalues should be strictly positive; limited
precision machine arithmetic may affect this.
Eigenvalues are often expressed as cumulative
percentages, representing the 'percentage variance
explained' by the associated axis or principal component.)

Eigenvectors:
(First three; their definition in terms of original vbes.)
0.0000 0.0000 0.0000
-0.5887 0.6815 -0.4347
-0.5478 0.0591 0.8345
-0.5945 -0.7294 -0.3385

Projections of row-points on first 3 prin. comps.:
-0.0755 -0.2148 -0.0133
0.3675 -0.0824 -0.0688
-0.1905 -0.2702 0.1398
-0.1893 0.0912 0.1150
-0.6852 -0.1403 -0.4468
0.0106 -0.0064 0.5619
-0.3015 0.5704 -0.0488
1.0639 0.0525 -0.2389

Projections of column-points on first 3 prin. comps.:
0.0000 0.0000 0.0000
-0.8126 0.4732 -0.3402
-0.7561 0.0410 0.6532
-0.8206 -0.5064 -0.2650*/

/*
Eigenvector matrix is used to convert original data set to projected value set.
This 4x3 matrix is used as (A^T)x=x' where x is 4 row vector and x' is output 3 row vector projected.
*/

//pca_simul2.cpp
/*********************************/
/* Principal Components Analysis */
/*********************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIGN(a, b) ((b) < 0 ? -fabs(a) : fabs(a))

float **matrix(int n, int m);
float *vector(int n);
void free_matrix(float **mat, int n, int m);
void free_vector(float *v, int n);
void corcol(float **data, int n, int m, float **symmat);
void covcol(float **data, int n, int m, float **symmat);
void scpcol(float **data, int n, int m, float **symmat);
void errhand(char err_msg[]);
void tred2(float **a, int n, float *d, float *e);
void tqli(float d[], float e[], int n, float **z);

int main(int argc, char *argv[])
{
    FILE *stream;
    int n, m, i, j, k, k2;
    float **data, **symmat, **symmat2, *evals, *interm;
    float in_value;
    double in_value_d;
    char option;

    /*********************************************************************
    Get from command line:
    input data file name, #rows, #cols, option.

        Open input file: fopen opens the file whose name is stored in the
        pointer argv[argc-1]; if unsuccessful, error message is printed to
        stderr.
    *********************************************************************/

    if(argc != 5) {
        printf("Syntax help: PCA filename #rows #cols option\n\n");
        printf("(filename -- give full path name,\n");
        printf(" #rows                          \n");
        printf(" #cols    -- integer values,\n");
        printf(" option   -- R (recommended) for correlation analysis,\n");
        printf("             V for variance/covariance analysis\n");
        printf("             S for SSCP analysis.)\n");
        exit(1);
    }

    n = atoi(argv[2]);                /* # rows */
    m = atoi(argv[3]);             /* # columns */
    option = argv[4][0];     /* Analysis option */

    printf("No. of rows: %d, no. of columns: %d.\n", n, m);
    printf("Input file: %s.\n", argv[1]);
    printf("IAMFINE\n");

    if((stream = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Program %s : cannot open file %s\n", argv[0], argv[1]);
        fprintf(stderr, "Exiting to system.");
        exit(1);
        /* Note: in versions of DOS prior to 3.0, argv[0] contains the
        string "C". */
    }

    /* Now read in data. */
    printf("STILLFINE\n");
    data = matrix(n, m);  /* Storage allocation for input data */
    printf("DATASTORED\n");

    for(i=1; i<=n; i++) {
        for(j=1; j<=m; j++) {
            fscanf(stream, "%lf,", &in_value_d); // data form
            in_value = (float) in_value_d;
            //printf("%d %d : %f\n", i, j, in_value);
            data[i][j] = in_value;
        }
    }

    /* Check on (part of) input data.
    for (i = 1; i <= 18; i++) {
        for (j = 1; j <= 8; j++) {
            printf("%7.1f", data[i][j]);
        }
        printf("\n");
    }*/

    symmat = matrix(m, m);  /* Allocation of correlation (etc.) matrix */

    /* Look at analysis option; branch in accordance with this. */
    switch(option) {
        case 'R':
        case 'r':
            printf("Analysis of correlations chosen.\n");
            corcol(data, n, m, symmat);

            /* Output correlation matrix.
            for (i = 1; i <= m; i++) {
                for (j = 1; j <= 8; j++) {
                printf("%7.4f", symmat[i][j]);
                }
                printf("\n");
            }*/
            break;

        case 'V':
        case 'v':
            printf("Analysis of variances-covariances chosen.\n");
            covcol(data, n, m, symmat);

            /* Output variance-covariance matrix.
            for (i = 1; i <= m; i++) {
                for (j = 1; j <= 8; j++) {
                    printf("%7.1f", symmat[i][j]);
                }
                printf("\n");
            }*/
            break;

        case 'S':
        case 's':
            printf("Analysis of sums-of-squares-cross-products");
            printf(" matrix chosen.\n");
            scpcol(data, n, m, symmat);

            /* Output SSCP matrix.
            for (i = 1; i <= m; i++) {
                for (j = 1; j <= 8; j++) {
                    printf("%7.1f", symmat[i][j]);
                }
                printf("\n");
            }*/
            break;

        default:
            printf("Option: %s\n",option);
            printf("For option, please type R, V, or S\n");
            printf("(upper or lower case).\n");
            printf("Exiting to system.\n");
            exit(1);
            break;
    }

    /*********************************************************************
    Eigen-reduction
    **********************************************************************/

    /* Allocate storage for dummy and new vectors. */
    evals = vector(m);     /* Storage alloc. for vector of eigenvalues */
    interm = vector(m);    /* Storage alloc. for 'intermediate' vector */
    symmat2 = matrix(m, m);  /* Duplicate of correlation (etc.) matrix */
    for(i=1; i<=m; i++) {
        for(j=1; j<=m; j++) {
            symmat2[i][j] = symmat[i][j]; /* Needed below for col. projections */
        }
    }

    tred2(symmat, m, evals, interm);  /* Triangular decomposition */
    tqli(evals, interm, m, symmat);   /* Reduction of sym. trid. matrix */
    /* evals now contains the eigenvalues,
    columns of symmat now contain the associated eigenvectors. */

    printf("\nEigenvalues:\n");
    for(j=m; j>=1; j--) {
        printf("%18.5f\n", evals[j]);
    }
    printf("\n(Eigenvalues should be strictly positive; limited\n");
    printf("precision machine arithmetic may affect this.\n");
    printf("Eigenvalues are often expressed as cumulative\n");
    printf("percentages, representing the 'percentage variance\n");
    printf("explained' by the associated axis or principal component.)\n");
    printf("\nEigenvectors:\n");
    printf("(First three; their definition in terms of original vbes.)\n");
    for(j=1; j<=m; j++) {
        for(i=1; i<=3; i++) {
            printf("%12.4f", symmat[j][m-i+1]);
        }
        printf("\n");
    }

    /* Form projections of row-points on first three prin. components. */
    /* Store in 'data', overwriting original data. */
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= m; j++) {
            interm[j] = data[i][j];
        }   /* data[i][j] will be overwritten */
        for (k = 1; k <= 3; k++) {
            data[i][k] = 0.0;
            for (k2 = 1; k2 <= m; k2++) {
                data[i][k] += interm[k2] * symmat[k2][m-k+1];
            }
        }
    }

    printf("\nProjections of row-points on first 3 prin. comps.:\n");
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= 3; j++) {
            printf("%12.4f", data[i][j]);
        }
        printf("\n");
    }

    /* Form projections of col.-points on first three prin. components. */
    /* Store in 'symmat2', overwriting what was stored in this. */
    for (j = 1; j <= m; j++) {
        for (k = 1; k <= m; k++) {
            interm[k] = symmat2[j][k];
        }  /*symmat2[j][k] will be overwritten*/
        for (i = 1; i <= 3; i++) {
            symmat2[j][i] = 0.0;
            for (k2 = 1; k2 <= m; k2++) {
                symmat2[j][i] += interm[k2] * symmat[k2][m-i+1];
            }
            if (evals[m-i+1] > 0.0005)   /* Guard against zero eigenvalue */
                symmat2[j][i] /= sqrt(evals[m-i+1]);   /* Rescale */
            else
                symmat2[j][i] = 0.0;    /* Standard kludge */
        }
    }

    printf("\nProjections of column-points on first 3 prin. comps.:\n");
    for (j = 1; j <= m; j++) {
        for (k = 1; k <= 3; k++) {
            printf("%12.4f", symmat2[j][k]);
        }
        printf("\n");
    }

    free_matrix(data, n, m);
    free_matrix(symmat, m, m);
    free_matrix(symmat2, m, m);
    free_vector(evals, m);
    free_vector(interm, m);

    return 0;
}

/**  Correlation matrix: creation  ***********************************/
void corcol(float **data, int n, int m, float **symmat)
/* Create m * m correlation matrix from given n * m data matrix. */
{
    float eps = 0.005;
    float x, *mean, *stddev;
    int i, j, j1, j2;

    /* Allocate storage for mean and std. dev. vectors */

    mean = vector(m);
    stddev = vector(m);

    /* Determine mean of column vectors of input data matrix */

    for(j=1; j<=m; j++) {
        mean[j] = 0.0;
        for(i=1; i<=n; i++) {
            mean[j] += data[i][j];
        }
        mean[j] /= (float) n;
    }

    printf("\nMeans of column vectors:\n");
    for(j=1; j<=m; j++) {
        printf("%7.1f", mean[j]);
    }
    printf("\n");

    /* Determine standard deviations of column vectors of data matrix. */

    for(j=1; j<=m; j++) {
        stddev[j] = 0.0;
        for(i=1; i<=n; i++) {
            stddev[j] += ((data[i][j] - mean[j]) * (data[i][j] - mean[j]));
        }
        stddev[j] /= (float) n;
        stddev[j] = sqrt(stddev[j]);
        /* The following in an inelegant but usual way to handle
        near-zero std. dev. values, which below would cause a zero-
        divide. */
        if(stddev[j] <= eps) stddev[j] = 1.0;
    }

    printf("\nStandard deviations of columns:\n");
    for(j=1; j<=m; j++) {
        printf("%7.1f", stddev[j]);
    }
    printf("\n");

    /* Center and reduce the column vectors. */

    for(i=1; i<=n; i++) {
        for(j=1; j<=m; j++) {
            data[i][j] -= mean[j];
            x = sqrt((float) n);
            x *= stddev[j];
            data[i][j] /= x;
        }
    }

    /* Calculate the m * m correlation matrix. */
    for(j1=1; j1<=m-1; j1++) {
        symmat[j1][j1] = 1.0;
        for(j2=j1+1; j2<=m; j2++) {
            symmat[j1][j2] = 0.0;
            for(i=1; i<=n; i++) {
                symmat[j1][j2] += (data[i][j1] * data[i][j2]);
            }
            symmat[j2][j1] = symmat[j1][j2];
        }
    }
    symmat[m][m] = 1.0;

    return;
}

/**  Variance-covariance matrix: creation  *****************************/
void covcol(float **data, int n, int m, float **symmat)/* Create m * m covariance matrix from given n * m data matrix. */
{
    float *mean;
    int i, j, j1, j2;
    
    /* Allocate storage for mean vector */

    mean = vector(m);

    /* Determine mean of column vectors of input data matrix */

    for (j = 1; j <= m; j++) {
        mean[j] = 0.0;
        for (i = 1; i <= n; i++) {
            mean[j] += data[i][j];
        }
        mean[j] /= (float)n;
    }

    printf("\nMeans of column vectors:\n");
    for (j = 1; j <= m; j++) {
        printf("%7.1f",mean[j]);
    }
    printf("\n");
    
    /* Center the column vectors. */
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= m; j++) {
            data[i][j] -= mean[j];
        }
    }

    /* Calculate the m * m covariance matrix. */
    for (j1 = 1; j1 <= m; j1++) {
        for (j2 = j1; j2 <= m; j2++) {
            symmat[j1][j2] = 0.0;
            for (i = 1; i <= n; i++) {
                symmat[j1][j2] += data[i][j1] * data[i][j2];
            }
            symmat[j2][j1] = symmat[j1][j2];
        }
    }

    return;
}

/**  Sums-of-squares-and-cross-products matrix: creation  **************/
void scpcol(float **data, int n, int m, float **symmat)/* Create m * m sums-of-cross-products matrix from n * m data matrix. */
{
    int i, j1, j2;

    /* Calculate the m * m sums-of-squares-and-cross-products matrix. */

    for (j1 = 1; j1 <= m; j1++) {
        for (j2 = j1; j2 <= m; j2++) {
            symmat[j1][j2] = 0.0;
            for (i = 1; i <= n; i++) {
                symmat[j1][j2] += data[i][j1] * data[i][j2];
            }
            symmat[j2][j1] = symmat[j1][j2];
        }
    }

    return;
}

/**  Error handler  **************************************************/
//char err_msg[];
void errhand(char err_msg[])/* Error handler */
{
    fprintf(stderr,"Run-time error:\n");
    fprintf(stderr,"%s\n", err_msg);
    fprintf(stderr,"Exiting to system.\n");
    exit(1);
}

/**  Allocation of vector storage  ***********************************/
float *vector(int n)/* Allocates a float vector with range [1..n]. */
{
    float *v;

    v = (float *) malloc ((unsigned) n*sizeof(float));
    if (!v) errhand("Allocation failure in vector().");

    return v-1;
}

/**  Allocation of float matrix storage  *****************************/
float **matrix(int n, int m)/* Allocate a float matrix with range [1..n][1..m]. */
{
    int i;
    float **mat;

    printf("A");
    /* Allocate pointers to rows. */
    mat = (float **) malloc((unsigned) n * sizeof(float **));
    if(!mat) errhand("Allocation failure 1 in matrix().");
    mat -= 1;

    /* Allocate rows and set pointers to them. */
    for(i=1; i<=n; i++) {
        mat[i] = (float *) malloc((unsigned) m * sizeof(float *));
        if(!mat[i]) errhand("Allocation failure 2 in matrix().");
        mat[i] -= 1;
    }

    /* Return pointer to array of pointers to rows. */
    return mat;
}

/**  Deallocate vector storage  *********************************/
void free_vector(float *v,int n)/* Free a float vector allocated by vector(). */
{
    free((char*) (v+1));
}

/**  Deallocate float matrix storage  ***************************/
void free_matrix(float **mat,int n,int m)/* Free a float matrix allocated by matrix(). */
{
    int i;

    for (i = n; i >= 1; i--) {
        free ((char*) (mat[i]+1));
    }
    free ((char*) (mat+1));
}

/**  Reduce a real, symmetric matrix to a symmetric, tridiag. matrix. */
void tred2(float **a, int n, float *d, float *e)
/* float **a, d[], e[]; */
/* Householder reduction of matrix a to tridiagonal form.
Algorithm: Martin et al., Num. Math. 11, 181-195, 1968.
Ref: Smith et al., Matrix Eigensystem Routines -- EISPACK Guide
Springer-Verlag, 1976, pp. 489-494.
W H Press et al., Numerical Recipes in C, Cambridge U P,
1988, pp. 373-374.  */
{
    int l, k, j, i;
    float scale, hh, h, g, f;
    
    for(i=n; i>=2; i--) {
        l = i - 1;
        h = scale = 0.0;
        if(l > 1) {
            for(k=1; k<=l; k++) scale += fabs(a[i][k]);
            if(scale == 0.0) e[i] = a[i][l];
            else {
                for(k=1; k<=l; k++) {
                    a[i][k] /= scale;
                    h += a[i][k] * a[i][k];
                }
                f = a[i][l];
                g = f > 0 ? -sqrt(h) : sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                a[i][l] = f - g;
                f = 0.0;
                for(j=1; j<=l; j++) {
                    a[j][i] = a[i][j] / h;
                    g = 0.0;
                    for(k=1; k<=j; k++) g += a[j][k] * a[i][k];
                    for(k=j+1; k<=l; k++) g += a[k][j] * a[i][k];
                    e[j] = g / h;
                    f += e[j] * a[i][j];
                }
                hh = f / (h + h);
                for(j=1; j<=l; j++) {
                    f = a[i][j];
                    e[j] = g = e[j] - hh * f;
                    for(k=1; k<=j; k++) a[j][k] -= (f * e[k] + g * a[i][k]);
                }
            }
        }
        else e[i] = a[i][l];
        d[i] = h;
    }
    d[1] = 0.0;
    e[1] = 0.0;
    for(i=1; i<=n; i++) {
        l = i - 1;
        if(d[i]) {
            for(j=1; j<=l; j++) {
                g = 0.0;
                for(k=1; k<=l; k++) g += a[i][k] * a[k][j];
                for(k=1; k<=l; k++) a[k][j] -= g * a[k][i];
            }
        }
        d[i] = a[i][i];
        a[i][i] = 1.0;
        for(j=1; j<=l; j++) a[j][i] = a[i][j] = 0.0;
    }
}

/**  Tridiagonal QL algorithm -- Implicit  **********************/
void tqli(float d[], float e[], int n, float **z)
{
    int m, l, iter, i, k;
    float s, r, p, g, f, dd, c, b;
    for(i=2; i<=n; i++) e[i-1] = e[i];
    e[n] = 0.0;
    for(l=1; l<=n; l++) {
        iter = 0;
        do {
            for(m=l; m<=n-1; m++) {
                dd = fabs(d[m]) + fabs(d[m+1]);
                if(fabs(e[m]) + dd == dd) break;
            }
            if(m != l) {
                if(iter++ == 30) errhand("No convergence in TLQI.");
                g = (d[l+1] - d[l]) / (2.0 * e[l]);
                r = sqrt((g * g) + 1.0);
                g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
                s = c = 1.0;
                p = 0.0;
                for(i=m-1; i>=l; i--) {
                    f = s * e[i];
                    b = c * e[i];
                    if(fabs(f) >= fabs(g)) {
                        c = g / f;
                        r = sqrt((c * c) + 1.0);
                        e[i+1] = f * r;
                        c *= (s = 1.0 / r);
                    }
                    else {
                        s = f / g;
                        r = sqrt((s * s) + 1.0);
                        e[i+1] = g * r;
                        s *= (c = 1.0/r);
                    }
                    g = d[i+1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;d[i+1] = g + p;
                    g = c * r - b;
                    for (k = 1; k <= n; k++) {
                        f = z[k][i+1];
                        z[k][i+1] = s * z[k][i] + c * f;
                        z[k][i] = c * z[k][i] - s * f;
                    } 
                }
                d[l] = d[l] - p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
}