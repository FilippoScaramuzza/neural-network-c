#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float w1;
    float w2;
    float b;
} Neuron;

typedef struct {
    Neuron n1;
    Neuron n2;
    Neuron n3;
} Network;

typedef float train[3];

train training_set[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
};

#define TRAIN_SIZE (sizeof(training_set) / sizeof(train))

float sigf(float x)
{
    return 1 / (1 + expf(-x));
}

float forward(Network n, float x1, float x2)
{
    float y1 = sigf(n.n1.w1 * x1 + n.n1.w2 * x2 + n.n1.b);
    float y2 = sigf(n.n2.w1 * x1 + n.n2.w2 * x2 + n.n2.b);
    float y3 = sigf(n.n3.w1 * y1 + n.n3.w2 * y2 + n.n3.b);
    return y3;
}

float mse(Network n)
{
    float distances = 0.f;

    for (size_t i = 0; i < TRAIN_SIZE; i++)
    {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        float y = training_set[i][2];
        float y_ = forward(n, x1, x2);
        distances += (y - y_) * (y - y_);
    }
    
    return distances / TRAIN_SIZE;
}

Network gradient_descent(Network n, size_t iterations)
{
    float eps = 1e-3;
    float rate = 5e-1;
    
    for (size_t i = 0; i < iterations; i++)
    {
        float dw11 = (mse((Network){(Neuron){n.n1.w1 + eps, n.n1.w2, n.n1.b}, n.n2, n.n3}) - mse(n)) / eps;
        float dw12 = (mse((Network){(Neuron){n.n1.w1, n.n1.w2 + eps, n.n1.b}, n.n2, n.n3}) - mse(n)) / eps;
        float db1 = (mse((Network){(Neuron){n.n1.w1, n.n1.w2, n.n1.b + eps}, n.n2, n.n3}) - mse(n)) / eps;
        float dw21 = (mse((Network){n.n1, (Neuron){n.n2.w1 + eps, n.n2.w2, n.n2.b}, n.n3}) - mse(n)) / eps;
        float dw22 = (mse((Network){n.n1, (Neuron){n.n2.w1, n.n2.w2 + eps, n.n2.b}, n.n3}) - mse(n)) / eps;
        float db2 = (mse((Network){n.n1, (Neuron){n.n2.w1, n.n2.w2, n.n2.b + eps}, n.n3}) - mse(n)) / eps;
        float dw31 = (mse((Network){n.n1, n.n2, (Neuron){n.n3.w1 + eps, n.n3.w2, n.n3.b}}) - mse(n)) / eps;
        float dw32 = (mse((Network){n.n1, n.n2, (Neuron){n.n3.w1, n.n3.w2 + eps, n.n3.b}}) - mse(n)) / eps;
        float db3 = (mse((Network){n.n1, n.n2, (Neuron){n.n3.w1, n.n3.w2, n.n3.b + eps}}) - mse(n)) / eps;
        n.n2.w1 -= rate * dw21;
        n.n2.w2 -= rate * dw22;
        n.n2.b -= rate * db2;
        n.n1.w1 -= rate * dw11;
        n.n1.w2 -= rate * dw12;
        n.n1.b -= rate * db1;
        n.n3.w1 -= rate * dw31;
        n.n3.w2 -= rate * dw32;
        n.n3.b -= rate * db3;
    }
    
    return n;
}

void print_network(Network n)
{
    printf("n1: w1 = %f, w2 = %f, b = %f\n", n.n1.w1, n.n1.w2, n.n1.b);
    printf("n2: w1 = %f, w2 = %f, b = %f\n", n.n2.w1, n.n2.w2, n.n2.b);
    printf("n3: w1 = %f, w2 = %f, b = %f\n", n.n3.w1, n.n3.w2, n.n3.b);
    printf("MSE: %f\n", mse(n));

    for (size_t i = 0; i < TRAIN_SIZE; i++) 
    {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        float y = training_set[i][2];
        float y_ = forward(n, x1, x2);
        printf("%f ^ %f = %f (%f)\n", x1, x2, y_, y);
    }
}

int main(void)
{
    srand(42);

    Network n = {
        {(float)rand() / RAND_MAX, 
         (float)rand() / RAND_MAX, 
         (float)rand() / RAND_MAX},
        {(float)rand() / RAND_MAX,
         (float)rand() / RAND_MAX,
         (float)rand() / RAND_MAX},
        {(float)rand() / RAND_MAX, 
         (float)rand() / RAND_MAX, 
         (float)rand() / RAND_MAX}
    };

    printf("=== Before Gradient Descent ===\n");
    print_network(n);
    
    n = gradient_descent(n, 100*1000);

    printf("\n====  After Gradient Descent ===\n");
    print_network(n);

    return 0;
}
