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

#define MAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]

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

typedef struct {
    size_t *archi;
    size_t num_layers;
    Matrix *weights;
    Matrix *biases;
    Matrix *activations; // num_layers + 1 (input)
} NeuralNetwork;

#define ARRAY_LEN(arr) (sizeof((arr)) / sizeof((arr)[0]))
#define NN_INPUT(nn) ((nn).activations[0])
#define NN_OUTPUT(nn) ((nn).activations[(nn).num_layers])

// size_t archi[] = {2, 2, 1}

NeuralNetwork nn_alloc(size_t *archi, size_t num_layers);
void nn_rand(NeuralNetwork nn, float min, float max);
void nn_print(NeuralNetwork nn, char *name);
void nn_forward(NeuralNetwork nn);
float nn_mse(NeuralNetwork nn, Matrix train_in, Matrix train_out);
void nn_finite_diff(NeuralNetwork nn, NeuralNetwork grad, float eps, Matrix train_in, Matrix train_out);
void nn_gradient_descent(NeuralNetwork nn, float eps, float rate, Matrix train_in, Matrix train_out, size_t iterations);

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

NeuralNetwork nn_alloc(size_t archi[], size_t num_layers)
{
    size_t input_size = archi[0];
    NeuralNetwork nn;
    nn.archi = archi;
    nn.num_layers = num_layers;
    nn.weights = malloc(num_layers * sizeof(*nn.weights));
    nn.biases = malloc(num_layers * sizeof(*nn.biases));
    nn.activations = malloc((num_layers + 1) * sizeof(*nn.activations));
    
    nn.activations[0] = mat_alloc(1, input_size);

    for (size_t i = 1; i <= nn.num_layers; i++)
    {
        nn.weights[i - 1] = mat_alloc(nn.activations[i - 1].cols, archi[i]);
        nn.biases[i - 1] = mat_alloc(1, archi[i]);
        nn.activations[i] = mat_alloc(1, archi[i]);
    }

    return nn;
}

void nn_rand(NeuralNetwork nn, float min, float max)
{
    for (size_t i = 0; i < nn.num_layers; i++)
    {
        mat_rand(nn.weights[i], min, max);
        mat_rand(nn.biases[i], min, max);
    }
}

void nn_print(NeuralNetwork nn, char *name)
{
    printf("%s: [\n", name);
    for (size_t i = 0; i < nn.num_layers; i++)
    {
        mat_print(nn.weights[i], "weights");
        mat_print(nn.biases[i], "biases");
    }
    printf("]\n");
}

void nn_forward(NeuralNetwork nn)
{
    for (size_t i = 0; i < nn.num_layers; i++)
    {
        mat_dot(nn.activations[i + 1], nn.activations[i], nn.weights[i]);
        mat_sum(nn.activations[i + 1], nn.biases[i]);
        mat_sigf(nn.activations[i + 1]);
    }
}

float nn_mse(NeuralNetwork nn, Matrix train_in, Matrix train_out)
{
    assert(train_in.rows == train_out.rows);
    assert(train_in.cols == NN_INPUT(nn).cols);
    assert(train_out.cols == nn.activations[nn.num_layers].cols);

    float result = 0.0f;

    for (size_t i = 0; i < train_in.rows; i++)
    {
        Matrix x = mat_row(train_in, i);
        Matrix y = mat_row(train_out, i);
        mat_cpy(NN_INPUT(nn), x);

        nn_forward(nn);

        for (size_t j = 0; j < train_out.cols; j++)
        {
            float diff = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            result += diff * diff;
        }
    }

    return result / train_in.rows;
} 

void nn_finite_diff(NeuralNetwork nn, NeuralNetwork grad, float eps, Matrix train_in, Matrix train_out)
{
    float saved;
    float mse = nn_mse(nn, train_in, train_out);
    for (size_t i = 0; i < nn.num_layers; i++)
    {
        for (size_t j = 0; j < nn.weights[i].rows; j++)
        {
            for (size_t k = 0; k < nn.weights[i].cols; k++)
            {
                saved = MAT_AT(nn.weights[i], j, k);
                MAT_AT(nn.weights[i], j, k) += eps;
                float new_mse = nn_mse(nn, train_in, train_out);
                MAT_AT(grad.weights[i], j, k) = (new_mse - mse) / eps;
                MAT_AT(nn.weights[i], j, k) = saved;
            }
        }
        for (size_t j = 0; j < nn.biases[i].rows; j++)
        {
            for (size_t k = 0; k < nn.biases[i].cols; k++)
            {
                saved = MAT_AT(nn.biases[i], 0, k);
                MAT_AT(nn.biases[i], 0, k) += eps;
                float new_mse = nn_mse(nn, train_in, train_out);
                MAT_AT(grad.biases[i], 0, k) = (new_mse - mse) / eps;
                MAT_AT(nn.biases[i], 0, k) = saved;
            }
        }
    }
}

void nn_gradient_descent(NeuralNetwork nn, float eps, float rate, Matrix train_in, Matrix train_out, size_t iterations)
{
    NeuralNetwork grad = nn_alloc(nn.archi, nn.num_layers);
    for (size_t it = 0; it < iterations; it++)
    {
        nn_finite_diff(nn, grad, eps, train_in, train_out);

        for (size_t i = 0; i < nn.num_layers; i++)
        {
            for (size_t j = 0; j < nn.weights[i].rows; j++)
            {
                for (size_t k = 0; k < nn.weights[i].cols; k++)
                {
                    MAT_AT(nn.weights[i], j, k) -= rate * MAT_AT(grad.weights[i], j, k);
                }
            }
            for (size_t j = 0; j < nn.biases[i].rows; j++)
            {
                for (size_t k = 0; k < nn.biases[i].cols; k++)
                {
                    MAT_AT(nn.biases[i], j, k) -= rate * MAT_AT(grad.biases[i], j, k);
                }
            }
        }
    }
}

#endif // NN_IMPLEMENTATION
