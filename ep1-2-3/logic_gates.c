#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    float w1;
    float w2;
    float b;
} Neuron;

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

float forward(Neuron n, float x1, float x2)
{
    return sigf(n.w1 * x1 + n.w2 * x2 + n.b);
}

float mse(Neuron n)
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

Neuron gradient_descent(Neuron n, size_t iterations)
{
	float eps = 1e-3;
	float rate = 5e-1;
    
	for (size_t i = 0; i < iterations; i++)
	{
        float dw1 = (mse((Neuron){n.w1 + eps, n.w2, n.b}) - mse(n)) / eps;
        float dw2 = (mse((Neuron){n.w1, n.w2 + eps, n.b}) - mse(n)) / eps;
        float db = (mse((Neuron){n.w1, n.w2, n.b + eps}) - mse(n)) / eps;
        n.w1 -= rate * dw1;
        n.w2 -= rate * dw2;
        n.b -= rate * db;
	}

	return n;
}

void print_model(Neuron n)
{
    printf("W1: %f, W2: %f, MSE: %f\n", n.w1, n.w2, mse(n));
    for (size_t i = 0; i < TRAIN_SIZE; i++) 
    {
        printf("%f * %f + %f * %f = %f (%f)\n", training_set[i][0],
                                                n.w1, 
                                                training_set[i][1],
                                                n.w2, 
                                                forward(n, training_set[i][0], training_set[i][1]),
                                                training_set[i][2]);
    }
}

int main(void)
{
    srand(42);

    Neuron n;
    n.w1 = (float)rand() / RAND_MAX;
    n.w2 = (float)rand() / RAND_MAX;

    printf("=== Before Gradient Descent ===\n");
    print_model(n);

    n = gradient_descent(n, 100000);

    printf("\n===  After Gradient Descent ===\n");
    print_model(n);

    return 0;
}
