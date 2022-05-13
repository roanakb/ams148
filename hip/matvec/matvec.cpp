/* Matrix vector product for GPU
   Roanak Baviskar */

#include <iostream>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <ctime.h>

/* Use Matrix Class! */
#include "mat.h"
#include "submat.h"

// Thread block size
#define BLOCK_SIZE 64

//
