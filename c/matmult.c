#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define IDX(m, i, j)    (i * m->n + j)

typedef struct {
  int    m;
  int    n;
  double *elts;
} Matrix;

Matrix* create_matrix(int m, int n)
{
  Matrix *res;

  res = (Matrix*) malloc(sizeof(Matrix));

  res->m = m;
  res->n = n;

  res->elts = (double*) malloc(m * n * sizeof(double));

  return res;
}

Matrix* zero(int m, int n)
{
  Matrix *res;

  res = create_matrix(m, n);

  for (int i = 0; i < m * n; ++i) {
    res->elts[i] = 0.0;
  }

  return res;
}

void destroy_matrix(Matrix* m)
{
  if (m->elts != NULL) {
    free(m->elts);
    m->elts = NULL;
  }

  free(m);
}

Matrix* mat_mult(Matrix* m1, Matrix *m2)
{
  if (m1->n != m2->m)
    return NULL;

  Matrix *m3 = create_matrix(m1->m, m2->n);

  for (int i = 0; i < m3->m; ++i) {
    for (int j = 0; j < m3->n; ++j) {
      for (int p = 0; p < m1->n; ++p) {
        m3->elts[IDX(m3, i, j)] += m1->elts[IDX(m1, i, p)] * m2->elts[IDX(m2, p, j)];
      }
    }
  }

  return m3;
}

Matrix* gen_mat_1(int m, int n, double start, double inc)
{
  Matrix* res = create_matrix(m, n);
  double acc = start;

  for (int i = 0; i < m * n; ++i) {
    res->elts[i] = acc;
    acc += inc;
  }

  return res;
}

double bench_mat_mult(int size)
{
  Matrix *m1 = gen_mat_1(size, size, 0.0, 0.01);
  Matrix *m2 = gen_mat_1(size, size, 3.2, 0.02);

  struct timeval t1;
  struct timeval t2;

  double elapsed = 0.0;

  gettimeofday(&t1, NULL);
  Matrix *m3 = mat_mult(m1, m2);
  gettimeofday(&t2, NULL);

  elapsed = ((double) t2.tv_sec) + ((double) t2.tv_usec) * 1e-6;
  elapsed -= ((double) t1.tv_sec) + ((double) t1.tv_usec) * 1e-6;

  destroy_matrix(m1);
  destroy_matrix(m2);
  destroy_matrix(m3);

  return elapsed;
}

int main(void)
{
  int N = 1200;
  double t = bench_mat_mult(N);

  printf("gemm, N = %d, time: %5.3f s \n", N, t);

  return 0;
}
