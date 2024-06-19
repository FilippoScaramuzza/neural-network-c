#define NN_IMPLEMENTATION
#include "nn.h"
#include "time.h"

int main(void)
{
    srand(69);
    size_t archi[] = {2, 2, 1};
    NeuralNetwork nn = nn_alloc(archi, ARRAY_LEN(archi) - 1);
    nn_rand(nn, 0.0f, 1.0f);

    float train_set[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0
    };

    Matrix train_in = {
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .data = train_set
    };
    
    Matrix train_out = {
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .data = train_set + 2
    };

    printf("MSE BEFORE: %f\n", nn_mse(nn, train_in, train_out));

    nn_gradient_descent(nn, 10, train_in, train_out, 1000*1000);

    printf("MSE  AFTER: %f\n", nn_mse(nn, train_in, train_out));

    for (size_t i = 0; i < 4; i++)
    {
        mat_cpy(NN_INPUT(nn), mat_row(train_in, i));
        nn_forward(nn);
        printf("%f - %f: %f\n", MAT_AT(NN_INPUT(nn), 0, 0), MAT_AT(NN_INPUT(nn), 0, 1), MAT_AT(NN_OUTPUT(nn), 0, 0));
    }

    return 0;
}