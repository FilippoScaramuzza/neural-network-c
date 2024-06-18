#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *data;
} Matrix;

float rand_float(void);
float sigf(float x);

Matrix mat_alloc(size_t rows, size_t cols);
void mat_print(Matrix m, char *name);
void mat_rand(Matrix m, float min, float max);
void mat_fill(Matrix m, float val);
void mat_cpy(Matrix dst, Matrix src);
Matrix mat_row(Matrix m, size_t row);

void mat_dot(Matrix dst, Matrix a, Matrix b);
void mat_sum(Matrix dst, Matrix a);
void mat_sigf(Matrix a);

#define MAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]

#endif // NN_H

#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float)rand() / RAND_MAX;
}

float sigf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

Matrix mat_alloc(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = malloc(rows * cols * sizeof(*m.data));

    assert(m.data != NULL);

    return m;
}

void mat_print(Matrix m, char *name)
{
    printf("%s: [\n", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("    %f", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_rand(Matrix m, float min, float max)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float() * (max - min) + min;
        }
    }
}

void mat_fill(Matrix m, float val)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = val;
        }
    }
}

void mat_cpy(Matrix dst, Matrix src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

Matrix mat_row(Matrix m, size_t row) 
{
    return (Matrix){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .data = &MAT_AT(m, row, 0)
    };
}

void mat_sum(Matrix dst, Matrix a)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_dot(Matrix dst, Matrix a, Matrix b)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    assert(a.cols == b.rows);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = 0.0f;
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sigf(Matrix a)
{
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            MAT_AT(a, i, j) = sigf(MAT_AT(a, i, j));
        }
    }
}

#endif // NN_IMPLEMENTATION
