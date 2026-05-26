#include "stdio.h"
#include <stdint.h>

void pad_input_channel(
    size_t input_cols, 
    size_t input_rows, 
    size_t x_padding, 
    size_t y_padding, 
    const int8_t* input, 
    int8_t* output
);