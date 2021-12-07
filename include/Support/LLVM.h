#pragma once

// MLIR includes a lot of forward declarations of LLVM types, use them.
#include "mlir/Support/LLVM.h"

// Can not forward declare inline functions with default arguments, so we
// include the header directly.
#include "mlir/Support/LogicalResult.h"
