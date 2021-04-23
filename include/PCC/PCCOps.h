#pragma once
#include "PCC/PCCDialect.h"
#include "PCC/PCCTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "PCC/PCC.h.inc"