#define NN_IMPLEMENTATION
#include "nn.h"
#include "time.h"

typedef struct {
    Matrix x;
    
    Matrix w1;
    Matrix b1;
    Matrix a1;

    Matrix w2;
    Matrix b2;
    Matrix a2;

    Matrix y;
} Xor;

float train_set[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0
};

void forward(Xor xor) {
    mat_dot(xor.a1, xor.x, xor.w1);
    mat_sum(xor.a1, xor.b1);
    mat_sigf(xor.a1);
    
    mat_dot(xor.a2, xor.a1, xor.w2);
    mat_sum(xor.a2, xor.b2);
    mat_sigf(xor.a2);

    mat_cpy(xor.y, xor.a2);
}

float mse(Xor m, Matrix train_in, Matrix train_out) 
{
    assert(train_in.rows == train_out.rows);
    assert(train_in.cols == m.x.cols);
    assert(train_out.cols == m.y.cols);

    float result = 0.0f;

    for (size_t i = 0; i < train_in.rows; i++)
    {
        Matrix x = mat_row(train_in, i);
        Matrix y = mat_row(train_out, i);
        mat_cpy(m.x, x);

        forward(m);

        for (size_t j = 0; j < train_out.cols; j++)
        {
            float diff = MAT_AT(m.y, 0, j) - MAT_AT(y, 0, j);
            result += diff * diff;
        }
    }

    return result / train_in.rows;
} 

Xor xor_alloc() 
{
    Xor m;
    m.x = mat_alloc(1, 2);
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);
    m.y = mat_alloc(1, 1);

    mat_rand(m.w1, 0.0f, 1.0f);
    mat_rand(m.b1, 0.0f, 1.0f);
    mat_rand(m.w2, 0.0f, 1.0f);
    mat_rand(m.b2, 0.0f, 1.0f);

    return m;
}

void finite_difference(Xor m, Xor grad, float eps, Matrix train_in, Matrix train_out) 
{
    float loss = mse(m, train_in, train_out);
    float saved;

    for (size_t i = 0; i < m.w1.rows; i++)
    {
        for (size_t j = 0; j < m.w1.cols; j++)
        {
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(grad.w1, i, j) = (mse(m, train_in, train_out) - loss) / eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b1.rows; i++)
    {
        for (size_t j = 0; j < m.b1.cols; j++)
        {
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(grad.b1, i, j) = (mse(m, train_in, train_out) - loss) / eps;
            MAT_AT(m.b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.w2.rows; i++)
    {
        for (size_t j = 0; j < m.w2.cols; j++)
        {
            saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(grad.w2, i, j) = (mse(m, train_in, train_out) - loss) / eps;
            MAT_AT(m.w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b2.rows; i++)
    {
        for (size_t j = 0; j < m.b2.cols; j++)
        {
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(grad.b2, i, j) = (mse(m, train_in, train_out) - loss) / eps;
            MAT_AT(m.b2, i, j) = saved;
        }
    }
}

void gradient_descent(Xor m, float rate, float eps, Matrix train_in, Matrix train_out, size_t iterations)
{
    Xor grad = xor_alloc();

    for (size_t i = 0; i < iterations; i++)
    {
        finite_difference(m, grad, eps, train_in, train_out);

        for (size_t j = 0; j < m.w1.rows; j++)
        {
            for (size_t k = 0; k < m.w1.cols; k++)
            {
                MAT_AT(m.w1, j, k) -= rate * MAT_AT(grad.w1, j, k);
            }
        }

        for (size_t j = 0; j < m.b1.rows; j++)
        {
            for (size_t k = 0; k < m.b1.cols; k++)
            {
                MAT_AT(m.b1, j, k) -= rate * MAT_AT(grad.b1, j, k);
            }
        }

        for (size_t j = 0; j < m.w2.rows; j++)
        {
            for (size_t k = 0; k < m.w2.cols; k++)
            {
                MAT_AT(m.w2, j, k) -= rate * MAT_AT(grad.w2, j, k);
            }
        }

        for (size_t j = 0; j < m.b2.rows; j++)
        {
            for (size_t k = 0; k < m.b2.cols; k++)
            {
                MAT_AT(m.b2, j, k) -= rate * MAT_AT(grad.b2, j, k);
            }
        }
    }
}

int main(void) 
{
    Matrix train_in = {
        .cols = 2,
        .rows = 4,
        .stride = 3,
        .data = train_set
    };

    Matrix train_out = {
        .cols = 1,
        .rows = 4,
        .stride = 3,
        .data = train_set + 2
    };
    
    srand(time(NULL));

    Xor xor = xor_alloc();
    
    printf("MSE BEFORE: %f\n", mse(xor, train_in, train_out));
    gradient_descent(xor, 5e-1, 1e-1, train_in, train_out, 10*1000);
    printf("MSE  AFTER: %f\n", mse(xor, train_in, train_out));

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            MAT_AT(xor.x, 0, 0) = i;
            MAT_AT(xor.x, 0, 1) = j;

            forward(xor);

            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(xor.y, 0, 0));
        }
    }

    return 0;
}