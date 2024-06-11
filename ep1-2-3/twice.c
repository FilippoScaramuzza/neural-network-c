#include <stdio.h>
#include <stdlib.h>	
#include <time.h>
#include <unistd.h>

typedef float train[2];

train training_set[] = {
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8}
};

#define TRAIN_SIZE (sizeof(training_set) / sizeof(train)) // amount of elements


float mse(float w)
{
	float distances = 0.f;

	for (size_t i = 0; i < TRAIN_SIZE; i++)
	{
		float x = training_set[i][0];
		float y = training_set[i][1];
		float y_ = x * w;
		distances += (y - y_) * (y - y_);
	}
	
	return distances / TRAIN_SIZE;
}

float gradient_descent(float w, size_t iterations)
{
	float eps = 1e-3;
	float rate = 1e-3;
	for (size_t i = 0; i < iterations; i++)
	{
		float dw = (mse(w + eps) - mse(w)) / eps;
		w -= rate * dw;
	}

	return w;
}

int main(void)
{
	srand(42);

	float w = (float)rand() / RAND_MAX;
	printf("=== Before Gradient Descent ===\n");
	printf("W: %f, MSE: %f\n", w, mse(w));
	for (size_t i = 0; i < TRAIN_SIZE; i++) 
	{
		printf("%f * %f = %f (%f)\n", training_set[i][0], w,  training_set[i][0] * w, training_set[i][1]);
	}

	w = gradient_descent(w, 1000);
	
	printf("\n===  After Gradient Descent ===\n");
	printf("W: %f, MSE: %f\n", w, mse(w));
	for (size_t i = 0; i < TRAIN_SIZE; i++) 
	{
		printf("%f * %f = %f (%f)\n", training_set[i][0], w,  training_set[i][0] * w, training_set[i][1]);
	}
	return 0;
}
